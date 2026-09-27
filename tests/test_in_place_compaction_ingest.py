"""In-place compaction must not re-store history the plugin already persisted.

Hermes' in-place compaction keeps the session id and then notifies the context
engine with ``on_session_start(sid, boundary_reason="compression",
old_session_id=sid)``. Production showed two replay paths after that edge:

* the same-id notification took the generic "new session" path, zeroed the
  ingest cursor, and the next ingest re-stored the fresh tail (the summary
  anchor made the cursor reconciliation fall back to storing the whole batch);
* a post-turn hook that delivered the pre-compaction transcript after compress()
  had already shortened the cursor re-stored the uncompressed suffix.
"""

import json

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "inplace-session"
LANE = "agent:main:telegram:group:1:1"


def _mock_summary(**_kwargs):
    return "Older work summary.\nExpand for details about: older work", 1


def _turns(start: int, count: int) -> list[dict]:
    messages: list[dict] = []
    for n in range(start, start + count):
        call_id = f"call_{n}"
        messages.extend([
            {"role": "user", "content": f"request {n}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": call_id, "type": "function",
                                "function": {"name": "terminal", "arguments": f'{{"n": {n}}}'}}],
            },
            {"role": "tool", "tool_call_id": call_id, "content": f"tool output {n}"},
            {"role": "assistant", "content": f"answer {n}"},
        ])
    return messages


def _duplicates(engine: LCMEngine) -> int:
    rows = engine._store.get_session_messages(SESSION)
    keys = [
        (
            row["role"],
            row.get("content") or "",
            row.get("tool_call_id") or "",
            json.dumps(row.get("tool_calls") or "", sort_keys=True),
        )
        for row in rows
    ]
    return len(keys) - len(set(keys))


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _mock_summary)
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "inplace.db"),
    )
    instance = LCMEngine(config=config)
    instance.on_session_start(SESSION, platform="telegram", conversation_id=LANE, context_length=200000)
    try:
        yield instance
    finally:
        instance.shutdown()


def _compact_in_place(engine: LCMEngine, history: list[dict]) -> list[dict]:
    """Mirror the host: preflight ingest, compress, commit, in-place notification.

    Hermes' in-place commit calls ``commit_memory_session(messages)``, which
    reaches ``on_session_end(sid, pre_compaction_messages)`` before the
    compaction boundary is announced with the same session id.
    """
    engine.should_compress_preflight(history)
    compressed = engine.compress(history)
    assert len(compressed) < len(history)
    engine.on_session_end(SESSION, history)
    engine.on_session_start(
        SESSION,
        boundary_reason="compression",
        old_session_id=SESSION,
        platform="telegram",
        conversation_id=LANE,
    )
    return compressed


def test_same_id_compression_boundary_keeps_cursor_on_compacted_context(engine):
    history = [{"role": "system", "content": "You are concise."}] + _turns(0, 12)
    engine.ingest(history)
    stored_before = engine._store.get_session_count(SESSION)

    compressed = _compact_in_place(engine, history)

    # The host continues from the compacted context and appends one new turn.
    engine.ingest(compressed + _turns(100, 1))

    assert _duplicates(engine) == 0
    assert engine._store.get_session_count(SESSION) == stored_before + 4


def test_late_pre_compaction_transcript_after_in_place_compaction_is_not_restored(engine):
    history = [{"role": "system", "content": "You are concise."}] + _turns(0, 12)
    engine.ingest(history)
    stored_before = engine._store.get_session_count(SESSION)

    _compact_in_place(engine, history)

    # A post-turn hook that still holds the uncompressed transcript fires after
    # the in-place commit. Every message in it is already durable.
    engine.ingest(history)

    assert _duplicates(engine) == 0
    assert engine._store.get_session_count(SESSION) == stored_before


def test_late_pre_compaction_transcript_with_new_tail_stores_only_new_messages(engine):
    history = [{"role": "system", "content": "You are concise."}] + _turns(0, 12)
    engine.ingest(history)
    stored_before = engine._store.get_session_count(SESSION)

    _compact_in_place(engine, history)
    engine.ingest(history + _turns(200, 1))

    assert _duplicates(engine) == 0
    assert engine._store.get_session_count(SESSION) == stored_before + 4


def test_repeated_in_place_compactions_stay_duplicate_free(engine):
    history = [{"role": "system", "content": "You are concise."}] + _turns(0, 12)
    engine.ingest(history)
    total_new = 0
    for round_start in (300, 400, 500):
        compressed = _compact_in_place(engine, history)
        history = compressed + _turns(round_start, 10)
        engine.ingest(history)
        total_new += 40

    assert _duplicates(engine) == 0
    assert engine._store.get_session_count(SESSION) == 1 + 48 + total_new


def test_genuine_repeated_content_after_compaction_is_still_stored(engine):
    history = [{"role": "system", "content": "You are concise."}] + _turns(0, 12)
    engine.ingest(history)
    compressed = _compact_in_place(engine, history)
    stored_before = engine._store.get_session_count(SESSION)

    # The user genuinely repeats an earlier message verbatim after compaction.
    repeat = {"role": "user", "content": "request 11"}
    engine.ingest(compressed + [repeat])

    rows = engine._store.get_session_messages(SESSION)
    assert engine._store.get_session_count(SESSION) == stored_before + 1
    assert rows[-1]["content"] == "request 11"


def _fresh_engine(tmp_path) -> LCMEngine:
    """A new runtime on the same database: gateway restart or agent rebuild."""
    config = LCMConfig(
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        database_path=str(tmp_path / "inplace.db"),
    )
    instance = LCMEngine(config=config)
    instance.on_session_start(SESSION, platform="telegram", conversation_id=LANE, context_length=200000)
    return instance


def test_fresh_runtime_resuming_compacted_context_stores_only_new_messages(engine, tmp_path):
    history = [{"role": "system", "content": "You are concise."}] + _turns(0, 12)
    engine.ingest(history)
    compressed = _compact_in_place(engine, history)
    later = compressed + _turns(600, 2)
    engine.ingest(later)
    stored_before = engine._store.get_session_count(SESSION)

    resumed = _fresh_engine(tmp_path)
    try:
        resumed.ingest(later + _turns(700, 1))
        assert resumed._store.get_session_count(SESSION) == stored_before + 4
        assert _duplicates(resumed) == 0
    finally:
        resumed.shutdown()


def test_fresh_runtime_recovers_messages_lost_before_restart(engine, tmp_path):
    history = [{"role": "system", "content": "You are concise."}] + _turns(0, 12)
    engine.ingest(history)
    compressed = _compact_in_place(engine, history)
    engine.ingest(compressed + _turns(800, 1))
    stored_before = engine._store.get_session_count(SESSION)

    # The next turn never reached LCM (for example a locked session-end flush).
    lost = _turns(801, 1)
    resumed = _fresh_engine(tmp_path)
    try:
        resumed.ingest(compressed + _turns(800, 1) + lost + _turns(802, 1))
        assert resumed._store.get_session_count(SESSION) == stored_before + 8
        assert _duplicates(resumed) == 0
    finally:
        resumed.shutdown()


def test_fresh_runtime_with_unrelated_summary_marker_keeps_existing_behavior(engine, tmp_path):
    history = [{"role": "system", "content": "You are concise."}] + _turns(0, 12)
    engine.ingest(history)
    stored_before = engine._store.get_session_count(SESSION)

    # A summary header that names no node of this session proves nothing.
    forged = {
        "role": "user",
        "content": "[Recent Summary (d0, node 999999)]\nnot ours\n[Expand for details: nothing]",
    }
    resumed = _fresh_engine(tmp_path)
    try:
        resumed.ingest([forged] + _turns(900, 1))
        assert resumed._store.get_session_count(SESSION) == stored_before + 4
    finally:
        resumed.shutdown()

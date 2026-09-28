"""Gateway restart must not re-store history LCM already persisted.

After a restart the host rebuilds each session from its own transcript and
hands it to a fresh engine. That list replays what LCM already stored, but not
byte-for-byte. Production showed five shapes that each broke the exact-suffix
proof, so the whole replay was persisted again on every restart:

* a tool result that quotes an externalized-payload placeholder (for example
  ``lcm_load_session`` output) was treated as that payload;
* LCM holds a user message inside a gateway-shutdown recovery note;
* LCM holds rows the host transcript lacks, or the reverse;
* the host keeps a text-only view of a multimodal tool result LCM stored as
  JSON parts;
* the host folded a preserved task list, or a later send, into a stored
  message.

The guard tests check the opposite: messages after the replay are stored, a
message LCM does not hold is never skipped, and a batch that does not replay
the durable tail is stored in full.
"""

import json

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "restart-replay-session"
LANE = "agent:main:telegram:dm:1"


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


@pytest.fixture
def config(tmp_path):
    return LCMConfig(database_path=str(tmp_path / "restart.db"))


def _engine(config) -> LCMEngine:
    engine = LCMEngine(config=config)
    engine.on_session_start(SESSION, platform="telegram", conversation_id=LANE, context_length=200000)
    return engine


def _close(engine: LCMEngine) -> None:
    engine._store.close()
    engine._dag.close()
    engine._lifecycle.close()


def _persist(config, messages: list[dict]) -> int:
    engine = _engine(config)
    engine._ingest_messages(messages)
    count = engine._store.get_session_count(SESSION)
    _close(engine)
    return count


def _restart_and_ingest(config, messages: list[dict]) -> tuple[LCMEngine, int]:
    engine = _engine(config)
    engine._ingest_messages(messages)
    return engine, engine._store.get_session_count(SESSION)


def _reason(engine: LCMEngine) -> str:
    return engine.get_status()["ingest_reconciliation"]["reason"]


# -- externalized placeholder quoted inside a tool result --------------------


def test_tool_result_quoting_an_externalized_placeholder_keeps_its_own_identity(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "restart.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
        large_output_externalization_path=str(tmp_path / "payloads"),
    )
    history = _turns(0, 3)
    history.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_big", "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"}}],
    })
    history.append({"role": "tool", "tool_call_id": "call_big", "content": "x" * 5000})
    stored = _persist(config, history)
    engine = _engine(config)
    placeholder = engine._store.get_session_tail(SESSION, limit=1)[0]["content"]
    assert placeholder.startswith("[Externalized tool output:")
    _close(engine)

    # A later tool (lcm_load_session, read_file of a transcript) returns text
    # that merely quotes the placeholder. It is ordinary content.
    quoting = {"role": "tool", "tool_call_id": "call_quote",
               "content": json.dumps({"messages": [{"content": placeholder}]})}
    history += [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_quote", "type": "function",
                         "function": {"name": "lcm_load_session", "arguments": "{}"}}]},
        quoting,
        {"role": "assistant", "content": "loaded"},
    ]
    stored = _persist(config, history)

    engine, count = _restart_and_ingest(config, list(history))
    try:
        assert count == stored
        rows = engine._store.get_session_messages(SESSION)
        assert sum(1 for row in rows if row.get("tool_call_id") == "call_quote") == 1
    finally:
        _close(engine)


# -- store-only rows ---------------------------------------------------------


def test_restart_replay_matches_a_recovery_note_around_the_user_message(config):
    history = _turns(0, 4)
    stored_history = history + [
        {"role": "user", "content": "[System note: The previous turn was interrupted by a gateway "
                                    "shutdown; the gateway is now back online.]\n\n[kc] "},
    ]
    host_history = history + [{"role": "user", "content": "[kc]"}]
    # LCM stored the recovery-wrapped turn; the host transcript kept the bare
    # message. Both sides then carry the same answer.
    stored = _persist(config, stored_history + [{"role": "assistant", "content": "back online"}])

    engine, count = _restart_and_ingest(config, host_history + [{"role": "assistant", "content": "back online"}])
    try:
        assert count == stored
        assert _reason(engine) == "aligned durable tail replay"
    finally:
        _close(engine)


def test_restart_replay_steps_over_rows_missing_from_host_transcript(config):
    history = _turns(0, 5)
    stored = _persist(config, history)
    # The host transcript lost one tool round trip that LCM stored live.
    host_history = history[:10] + history[12:]

    engine, count = _restart_and_ingest(config, host_history)
    try:
        assert count == stored
        assert _reason(engine) == "aligned durable tail replay"
    finally:
        _close(engine)


def test_host_rows_lcm_never_stored_are_persisted_not_skipped(config):
    history = _turns(0, 5)
    stored_history = history[:10] + history[12:]
    stored = _persist(config, stored_history)

    # The host holds a tool round trip LCM never stored. The replay before it
    # is skipped; from it onward the batch is persisted so nothing is lost.
    engine, count = _restart_and_ingest(config, history)
    try:
        assert count == stored + (len(history) - 10)
        rows = engine._store.get_session_messages(SESSION)
        assert any(row.get("tool_call_id") == "call_2" for row in rows)
        assert _reason(engine) == "aligned durable tail replay"
        assert engine._ingest_cursor == len(history)
    finally:
        _close(engine)


# -- equivalent content ------------------------------------------------------


def test_restart_replay_matches_text_view_of_stored_multimodal_tool_result(config):
    history = _turns(0, 4)
    multimodal = [
        {"type": "text", "text": "Image loaded into your context.\n\nQuestion: check the layout"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 64}},
    ]
    # Hermes persists the text parts and replaces each image with [screenshot].
    history += [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_img", "type": "function",
                         "function": {"name": "vision", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_img", "content": multimodal},
        {"role": "assistant", "content": "layout looks fine"},
    ]
    stored = _persist(config, history)
    host_history = [dict(m) for m in history]
    host_history[-2]["content"] = "Image loaded into your context.\n\nQuestion: check the layout\n[screenshot]"

    engine, count = _restart_and_ingest(config, host_history)
    try:
        assert count == stored
    finally:
        _close(engine)


def test_restart_replay_matches_host_folded_task_list(config):
    history = _turns(0, 4) + [{"role": "user", "content": "[kc] done"},
                              {"role": "assistant", "content": "noted"}]
    stored = _persist(config, history)
    host_history = [dict(m) for m in history]
    host_history[-2]["content"] = (
        "[kc] done\n\n[Your active task list was preserved across context compression]\n- [>] deploy"
    )

    engine, count = _restart_and_ingest(config, host_history)
    try:
        assert count == stored
    finally:
        _close(engine)


# -- loss guards -------------------------------------------------------------


def test_new_messages_after_an_aligned_replay_are_stored(config):
    history = _turns(0, 5)
    stored = _persist(config, history)
    host_history = history[:10] + history[12:]

    engine, count = _restart_and_ingest(config, host_history + _turns(50, 1))
    try:
        assert count == stored + 4
        rows = engine._store.get_session_messages(SESSION)
        assert [row["content"] for row in rows[-4:]][::3] == ["request 50", "answer 50"]
    finally:
        _close(engine)


def test_new_turn_repeating_the_stored_tail_is_still_stored(config):
    history = _turns(0, 4) + [{"role": "user", "content": "try again"},
                              {"role": "assistant", "content": "retrying"}]
    stored = _persist(config, history)

    # A later turn repeats the last exchange verbatim after an unrelated one.
    engine, count = _restart_and_ingest(
        config, history + _turns(60, 1) + [{"role": "user", "content": "try again"},
                                          {"role": "assistant", "content": "retrying"}]
    )
    try:
        assert count == stored + 6
    finally:
        _close(engine)


def test_immediate_repeat_of_the_last_exchange_is_still_stored(config):
    history = _turns(0, 4) + [{"role": "user", "content": "try again"},
                              {"role": "assistant", "content": "retrying"}]
    stored = _persist(config, history)

    # The user repeats the last exchange straight away. Aligning the repeat
    # with the stored originals would need two bridged rows; aligning the
    # originals needs none, so the repeat is new and must be stored.
    engine, count = _restart_and_ingest(
        config, history + [{"role": "user", "content": "try again"},
                           {"role": "assistant", "content": "retrying"}]
    )
    try:
        assert count == stored + 2
    finally:
        _close(engine)


def test_batch_that_does_not_reach_the_durable_end_is_not_skipped(config):
    history = _turns(0, 6)
    stored = _persist(config, history)
    # An older slice of the session followed by new work: the alignment cannot
    # reach the stored tail's final row, so nothing may be skipped as replay.
    replay = history[:8] + _turns(70, 1)

    engine, count = _restart_and_ingest(config, replay)
    try:
        assert count == stored + len(replay)
    finally:
        _close(engine)


def test_folded_text_must_extend_at_a_line_boundary(config):
    history = _turns(0, 4) + [{"role": "user", "content": "ship"},
                              {"role": "assistant", "content": "shipped"}]
    stored = _persist(config, history)
    host_history = [dict(m) for m in history]
    # "shipping" is a different message, not a fold of "ship".
    host_history[-2]["content"] = "shipping"
    host_history[-1]["content"] = "shipped again"

    engine, count = _restart_and_ingest(config, host_history)
    try:
        assert count > stored
    finally:
        _close(engine)


def test_too_many_one_sided_rows_fall_back_to_persisting(config):
    history = _turns(0, 6)
    stored = _persist(config, history)
    # Drop every other tool round trip: far more one-sided rows than a replay
    # of the durable tail can explain.
    host_history = [m for k, m in enumerate(history) if (k // 4) % 2 == 0 or k % 4 in (0, 3)]

    engine, count = _restart_and_ingest(config, host_history)
    try:
        assert count > stored
        assert _reason(engine) != "aligned durable tail replay"
    finally:
        _close(engine)



def test_unrecoverable_persisted_output_marker_is_left_to_the_exact_rules(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "restart.db"))
    marker = (
        "<persisted-output>\n"
        "This tool result was too large (100 characters, 0.1 KB).\n"
        f"Full output saved to: {tmp_path / 'gone' / 'result.txt'}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        "Preview (first 30 chars):\n"
        "SAVED_OUTPUT_PREVIEW\n...\n"
        "</persisted-output>"
    )
    history = _turns(0, 3) + [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_saved", "type": "function",
                         "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_saved", "content": marker},
    ] + _turns(10, 1)
    stored = _persist(config, history)

    # LCM stores such a marker with identity metadata and, when the saved file
    # is gone, deliberately persists a replay of it again (duplicate over
    # loss). Alignment must not override that rule.
    engine, count = _restart_and_ingest(config, list(history))
    try:
        assert count > stored
        assert _reason(engine) != "aligned durable tail replay"
    finally:
        _close(engine)

"""Real owner decisions, with synthetic delayed transport and isolated SQLite."""
import copy
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import decision_adapter as adapter
from hermes_lcm import tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore


class DelayedFacade:
    def negotiate(self, version):
        return {"version": version, "supported": True,
                "owner_capabilities": ["rank_candidates", "select_windows", "expand_one_owned_ref"]}

    def __init__(self, delay=.015, alter=None):
        self.delay, self.alter, self.requests = delay, alter, []

    def rank_candidates(self, request):
        self.requests.append(copy.deepcopy(request))
        time.sleep(self.delay)
        ids = [c["candidate_id"] for c in request["facts"]["candidates"]][::-1]
        answer = {"request_id": request["request_id"], "candidate_ids": ids, "conflict_ids": ids[:1]}
        return self.alter(answer) if self.alter else answer

    def expand_one_owned_ref(self, request):
        self.requests.append(copy.deepcopy(request))
        time.sleep(self.delay)
        answer = {"request_id": request["request_id"],
                  "candidate_id": request["facts"]["candidates"][-1]["candidate_id"],
                  "relation": "states_missing_decision"}
        return self.alter(answer) if self.alter else answer


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "owner.db"), embeddings_enabled=False)
    store, dag = MessageStore(config.database_path), SummaryDAG(config.database_path)
    value = SimpleNamespace(_config=config, _store=store, _dag=dag,
                            _hermes_home=str(tmp_path), current_session_id="current")
    yield value
    dag.close()
    store.close()


def entries():
    return [{"hit": {"kind": "message_excerpt", "store_id": i + 1,
                     "session_id": "current", "snippet": f"Claim {i}; qualifier retained.",
                     "role": "user", "source": "cli", "lineage": ["parent"],
                     "exact_ref": f"lcm:{i + 1}:0-4"},
             "rrf_score": .05 / (i + 1), "_final_score": .03 / (i + 1)} for i in range(10)]


def rank(facade, incoming, **kwargs):
    return tools._lcm_recall_rerank(
        kwargs.pop("provider", None), "claim", incoming, window=8,
        deadline=kwargs.pop("deadline", time.monotonic() + 1),
        config=SimpleNamespace(rerank_enabled=True), supervision=facade,
        scope={"session_scope": "current"}, completeness={"finite_coverage": False}, **kwargs)


def test_worker_startup_is_charged_to_same_deadline(monkeypatch):
    clock, joined = [100.0], []
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    class Worker:
        def __init__(self, *, target, **kwargs):
            self.target = target
        def start(self):
            clock[0] += .1
            self.target()
        def join(self, remaining):
            joined.append(remaining)
        def is_alive(self):
            return False
    monkeypatch.setattr(tools.threading, "Thread", Worker)
    assert tools._run_within_deadline(lambda: "value", remaining_s=.15, name="test") == "value"
    assert joined[0] == pytest.approx(.05)


def test_first_uncached_rank_changes_real_owner_within_admission():
    incoming, facade = entries(), DelayedFacade()
    before = copy.deepcopy(incoming)
    provider = SimpleNamespace(provider_id="voyage", rerank=lambda *a, **k: pytest.fail("double rank"))
    start = time.monotonic()
    result, status = rank(facade, incoming, provider=provider)
    assert .01 <= time.monotonic() - start < .150
    assert result[:8] == incoming[:8][::-1] and result[8:] == incoming[8:]
    assert all(any(item is original for original in incoming) for item in result)
    assert incoming == before and status == "applied"
    assert len(facade.requests) == 1
    assert facade.requests[0]["completeness"] == {"finite_coverage": False}


@pytest.mark.parametrize("alter", [lambda a: None, lambda a: {**a, "request_id": "stale"},
    lambda a: {**a, "candidate_ids": ["invented"]},
    lambda a: {**a, "candidate_ids": a["candidate_ids"][:-1]},
    lambda a: {**a, "candidate_ids": [a["candidate_ids"][0]] * 8},
    lambda a: {**a, "candidate_ids": [{}]},
    lambda a: {**a, "conflict_ids": ["foreign"]}])
def test_invalid_proposals_keep_exact_baseline(alter):
    incoming = entries()
    assert rank(DelayedFacade(delay=0, alter=alter), incoming)[0] is incoming


def test_expiration_has_no_late_effect_or_cached_hint():
    incoming, facade = entries(), DelayedFacade(.09)
    start = time.monotonic()
    result, _ = rank(facade, incoming, deadline=start + .03)
    assert time.monotonic() - start < .15 and result is incoming
    time.sleep(.10)
    assert result == entries()
    facade.alter = lambda _: None
    assert rank(facade, incoming)[0] is incoming


def test_absent_capability_retains_voyage_and_serialized_baseline():
    incoming = entries()
    provider = SimpleNamespace(provider_id="voyage", rerank=lambda q, d, **k: [(1, .9), (0, .1)])
    a = rank(None, incoming, provider=provider)
    b = rank(object(), incoming, provider=provider)
    inactive = DelayedFacade()
    inactive.negotiate = lambda version: {"version": version, "supported": True, "owner_capabilities": []}
    c = rank(inactive, incoming, provider=provider)
    assert json.dumps(a) == json.dumps(b) == json.dumps(c)
    assert not inactive.requests


def test_real_recall_consumer_preserves_exact_refs_roles_coverage(engine):
    for text in (f"Claim {i} accepted" for i in range(10)):
        engine._store.append("current", {"role": "user", "content": text})
    args = {"query": "Claim", "detail": "answer_ready", "limit": 3, "seen_refs": []}
    baseline = json.loads(tools.lcm_recall(args, engine=engine))
    engine.supervision = DelayedFacade()
    start = time.monotonic()
    actual = json.loads(tools.lcm_recall(args, engine=engine))
    assert time.monotonic() - start < .15
    assert actual["hits"][0]["store_id"] != baseline["hits"][0]["store_id"]
    assert all(x["exact_ref"].startswith(f"lcm:{x['store_id']}:") for x in actual["hits"])
    assert {x["role"] for x in actual["hits"]} == {"user"}
    assert actual["provenance"]["rerank"] == "applied"
    actual["provenance"]["rerank"] = baseline["provenance"]["rerank"]
    for key in ("coverage", "computation", "provenance"):
        assert actual.get(key) == baseline.get(key)


def test_runtime_clone_retains_facade_not_recovery_budget(tmp_path):
    from hermes_lcm.engine import LCMEngine

    original = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "clones.db")),
                         hermes_home=str(tmp_path))
    original.supervision = DelayedFacade()
    original._supervision_recovered_slots = {("current", "old")}
    clone = original.clone_for_agent()
    try:
        assert clone.supervision is original.supervision
        assert not getattr(clone, "_supervision_recovered_slots", set())
    finally:
        clone.shutdown()
        original.shutdown()


def test_real_recall_expired_decision_returns_baseline(engine):
    for text in ("Claim first", "Claim second"):
        engine._store.append("current", {"role": "user", "content": text})
    args = {"query": "Claim", "limit": 2}
    baseline = json.loads(tools.lcm_recall(args, engine=engine))
    engine.supervision = DelayedFacade(.12)
    engine._config.recall_query_timeout_s = .04
    start = time.monotonic()
    actual = json.loads(tools.lcm_recall(args, engine=engine))
    assert time.monotonic() - start < .15
    assert [h["store_id"] for h in actual["hits"]] == [h["store_id"] for h in baseline["hits"]]
    time.sleep(.13)
    assert actual["provenance"]["rerank"] == "disabled"


def slot(engine, session="current"):
    text = "User decided to keep the existing protocol."
    store_id = engine._store.append(session, {"role": "user", "content": text})
    return {"slot_id": "missing-protocol", "question": "What protocol was decided?", "expansion_budget": 1,
            "explicit_ref_available": False,
            "visible_refs": [], "hits": [{"exact_ref": f"lcm:{store_id}:0-{len(text)}", "excerpt": text,
                                          "current": True, "superseded": False}]}


def test_one_existing_history_expansion_no_retrieval(engine, monkeypatch):
    missing = slot(engine)
    engine.supervision = DelayedFacade()
    monkeypatch.setattr(tools, "lcm_recall", lambda *a, **k: pytest.fail("new retrieval"))
    original, calls = tools.lcm_expand, []
    def expand(args, **kwargs):
        calls.append(args)
        return original(args, **kwargs)
    monkeypatch.setattr(tools, "lcm_expand", expand)
    result = adapter.recover_missing_history(engine, missing, deadline=time.monotonic() + .15)
    assert result["content"] == missing["hits"][0]["excerpt"]
    assert result["role"] == "user" and result["session_id"] == "current"
    assert len(calls) == 1 and calls[0]["max_tokens"] == 600
    assert adapter.recover_missing_history(engine, missing, deadline=time.monotonic() + .15) is None
    assert len(calls) == 1


@pytest.mark.parametrize("problem", ["foreign", "visible", "fabricated", "budget", "unknown", "stale"])
def test_missing_history_never_widens_or_fabricates(engine, problem):
    missing = slot(engine, "foreign" if problem == "foreign" else "current")
    engine.supervision = DelayedFacade(0)
    if problem == "visible":
        missing["visible_refs"] = [missing["hits"][0]["exact_ref"]]
    if problem == "fabricated":
        missing["hits"][0]["excerpt"] = "fabricated decision"
    if problem == "budget":
        missing["expansion_budget"] = 0
    if problem == "unknown":
        engine.supervision.alter = lambda a: {**a, "candidate_id": "lcm:9999:0-3"}
    if problem == "stale":
        engine.supervision.alter = lambda a: {**a, "request_id": "old"}
    assert adapter.recover_missing_history(engine, missing, deadline=time.monotonic() + .15) is None


def test_exact_history_hook_consumer_and_absent_byte_parity(engine):
    spec = importlib.util.spec_from_file_location("hermes_lcm.owner_entry", Path(__file__).parents[1] / "__init__.py")
    entry = importlib.util.module_from_spec(spec)
    entry.__package__ = "hermes_lcm"
    spec.loader.exec_module(entry)
    missing = slot(engine)
    payload = {"missing_history_slot": missing, "supervision_deadline": time.monotonic() + .15}
    assert entry._pre_llm_context(engine, "policy", payload) == {"context": "policy"}
    engine.supervision = DelayedFacade()
    result = entry._pre_llm_context(engine, "policy", payload)
    assert missing["hits"][0]["excerpt"] in result["context"]
    assert entry._pre_llm_context(engine, "policy", {}) == {"context": "policy"}


@pytest.mark.parametrize("field", ["current", "superseded", "explicit_ref_available"])
def test_unknown_history_owner_facts_fail_before_decision(engine, field):
    missing = slot(engine)
    engine.supervision = DelayedFacade(0)
    if field == "explicit_ref_available":
        missing.pop(field)
    else:
        missing["hits"][0].pop(field)
    assert adapter.recover_missing_history(engine, missing, deadline=time.monotonic() + .15) is None
    assert not engine.supervision.requests


def test_small_rank_batch_is_not_a_triage_opportunity():
    incoming, facade = entries()[:3], DelayedFacade(0)
    assert rank(facade, incoming)[0] is incoming
    assert not facade.requests


def test_source_projection_hydrates_whole_native_row_not_fts_markup(engine):
    incoming = []
    for i in range(10):
        text = f"Claim {i}; qualifying limitation."
        sid = engine._store.append("current", {"role": "user", "content": text})
        incoming.append({"hit": {"kind": "message_excerpt", "store_id": sid, "session_id": "current",
                                  "role": "user", "snippet": f"[Claim] {i}"}, "_final_score": i})
    facade = DelayedFacade(0)
    result = adapter.rank_candidates(facade, "Claim", incoming, window=8, engine=engine,
                                     scope={"session_scope": "all"}, completeness={}, deadline=time.monotonic() + .15)
    assert result[:8] == incoming[:8][::-1] and result[8:] == incoming[8:]
    facts = facade.requests[0]["facts"]
    assert all(c["qualifiers_complete"] is True and c["constraints_match"] is True for c in facts["candidates"])
    assert all("qualifying limitation" in c["excerpt"] for c in facts["candidates"])
    assert incoming[0]["hit"]["snippet"] == "[Claim] 0"

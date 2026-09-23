"""Regressions for ownership, child isolation, coverage, and tool eligibility."""
import json
import os
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm import tokens
from .subprocess_sandbox import run_isolated_python


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "_encoder_ready", True)
    monkeypatch.setattr(tokens, "_encoder", None)
    instance = LCMEngine(LCMConfig(
        database_path=str(tmp_path / "evidence.db"), sqlite_journal_mode="delete",
    ))
    instance.on_session_start("audit-regression", platform="cli")
    try:
        yield instance
    finally:
        instance.shutdown()


def test_real_host_ledger_unload_disarms_postturn_and_closes_only_prototype(tmp_path):
    # Fresh interpreter: packaging tests deliberately stub host modules. Exercise
    # the actual host ledger/dispatch, never its singleton or plugin discovery.
    script = r'''
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
# Standalone plugin CI provides only the ContextEngine ABC, not Hermes's real
# registration ledger. Skip only an absent host, never a broken installed one.
if importlib.util.find_spec("hermes_cli") is None:
    print(json.dumps({"skip": "real Hermes host is not installed"}))
    raise SystemExit(0)
import hermes_cli.plugins as host
repo = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("lcm_real_ledger_probe", repo / "__init__.py", submodule_search_locations=[str(repo)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from lcm_real_ledger_probe import tokens
tokens._encoder_ready = True
tokens._encoder = None
manager = host.PluginManager(scope_key=os.environ["HERMES_HOME"])
ctx = host.PluginContext(host.PluginManifest(name="hermes-lcm", path=str(repo), source="user"), manager)
module.register(ctx)
prototype = manager._context_engine
connection = prototype._store._conn
clone = prototype.clone_for_agent()
clone.on_session_start("owned-turn", platform="cli")
prototype.on_session_start("prototype-turn", platform="cli")
captured = manager._hooks["post_llm_call"][0]
other_calls = []
other = host.PluginContext(host.PluginManifest(name="unrelated-test-owner", path=str(repo), source="user"), manager)
other.register_hook("post_llm_call", lambda **kw: other_calls.append(kw["session_id"]))
assert len(manager._hooks["post_llm_call"]) == 2
try:
    # A known non-LCM engine must veto even a matching live registry binding.
    for selected in (SimpleNamespace(name="compressor"), None):
        manager.invoke_hook("post_llm_call", context_compressor=selected,
            session_id="prototype-turn", conversation_history=[{"role":"user","content":"NON_LCM_CANARY"}])
    manager.invoke_hook("post_llm_call", session_id="unbound-turn",
        conversation_history=[{"role":"user","content":"UNBOUND_CANARY"}])
    assert prototype._store.get_session_count("prototype-turn") == 0
    assert prototype._store.get_session_count("unbound-turn") == 0
    history = [{"role":"user","content":"OWNED_BEFORE_UNLOAD"}]
    manager.invoke_hook("post_llm_call", session_id="owned-turn", conversation_history=history)
    assert clone._store.get_session_count("owned-turn") == 1
    assert manager.unload("hermes-lcm") is True
    assert manager._context_engine is None
    assert len(manager._hooks["post_llm_call"]) == 1  # unrelated owner preserved
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError("prototype connection survived unload")
    history.append({"role":"user","content":"AFTER_UNLOAD_CANARY"})
    manager.invoke_hook("post_llm_call", session_id="owned-turn", conversation_history=history)
    # A dispatcher may already have captured a callback before ledger disposal.
    captured(context_compressor=clone, session_id="owned-turn", conversation_history=history)
    assert clone._store.get_session_count("owned-turn") == 1
    assert other_calls[-1] == "owned-turn"
    print(json.dumps({"real_host": str(Path(host.__file__).resolve()), "unloaded": True,
        "prototype_closed": True, "clone_rows_after_unload": 1, "unrelated_hook_preserved": True}))
finally:
    clone.shutdown()
'''
    result = run_isolated_python(script, tmp_path, Path(__file__).resolve().parent.parent)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if payload.get("skip") == "real Hermes host is not installed":
        pytest.skip(payload["skip"])
    assert payload["unloaded"] and payload["prototype_closed"]
    assert payload["clone_rows_after_unload"] == 1
    assert payload["real_host"].endswith("/hermes_cli/plugins.py")


def test_child_preloads_same_sqlite_deny_guard_and_network_guard(tmp_path, monkeypatch):
    denied = tmp_path / "sacrificial-denied"
    denied.mkdir()
    monkeypatch.setenv("HERMES_TEST_DENY_SQLITE_ROOTS", os.pathsep.join(filter(None, [
        os.environ.get("HERMES_TEST_DENY_SQLITE_ROOTS", ""), str(denied),
    ])))
    script = r'''
import json
from pathlib import Path
import socket
import sqlite3
import sys
root = Path(sys.argv[1])
alias = Path.cwd() / "alias"
alias.symlink_to(root, target_is_directory=True)
blocked = 0
for target in (str(root / "direct.db"), (root / "uri.db").as_uri() + "?mode=rwc", str(alias / "linked.db")):
    try:
        sqlite3.connect(target, uri=target.startswith("file:"))
    except RuntimeError as exc:
        assert "protected Hermes home" in str(exc)
        blocked += 1
    else:
        raise AssertionError("child opened a denied database")
try:
    socket.getaddrinfo("example.invalid", 443)
except RuntimeError:
    blocked += 1
else:
    raise AssertionError("child attempted network access")
assert not list(root.iterdir())
print(json.dumps({"blocked": blocked}))
'''
    result = run_isolated_python(script, tmp_path / "child", denied)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"blocked": 4}
    assert list(denied.iterdir()) == []


@pytest.mark.parametrize("claimed_complete", [None, False, True])
def test_omitted_exact_city_cannot_be_certified_as_historical_total(engine, claimed_complete):
    engine.ingest([
        {"role": "user", "content": "Alice visited Paris."},
        {"role": "user", "content": "Alice visited Rome."},
    ])
    rows = engine._store._conn.execute("SELECT store_id, content FROM messages ORDER BY store_id").fetchall()
    operands = [dict(store_id=row[0], span_start=0, span_end=len(row[1]), quote=row[1], key=city, unit="item")
                for row, city in zip(rows, ["paris", "rome"])]
    args = {"question": "How many cities did Alice visit?", "operands": operands[:1]}
    if claimed_complete is not None:
        args["evidence_complete"] = claimed_complete
    subset = json.loads(engine.handle_tool_call("lcm_compute", args))
    assert subset["status"] == "subset_computed"
    assert subset["evidence_complete"] is False
    assert subset["trace"]["result_value"] == 1
    assert subset["answer"].startswith("Selected evidence only (historical coverage unverified):")
    assert subset["trace"]["answer"] == subset["answer"]
    assert subset["trace"]["scope"] == "selected_evidence_only"
    assert subset["provenance"]["stages"]["selector"]["evidence_complete"] is False
    # Even a numerically correct, cited candidate cannot smuggle a total claim.
    candidate = "1 " + " ".join(f"[{ref}]" for ref in subset["trace"]["citations"])
    replay = json.loads(engine.handle_tool_call("lcm_compute", {**args, "candidate_answer": candidate}))
    assert replay["candidate_verification"]["status"] == "fallback"
    assert replay["answer"] == subset["answer"]
    full_selection = json.loads(engine.handle_tool_call("lcm_compute", {**args, "operands": operands}))
    assert full_selection["status"] == "subset_computed"  # selected all isn't proof of all
    assert full_selection["trace"]["result_value"] == 2


@pytest.mark.parametrize("assertions,adaptive", [(False, False), (True, False), (False, True), (True, True)])
def test_one_descriptor_controls_both_schema_paths_and_dispatch(tmp_path, monkeypatch, assertions, adaptive):
    from .test_packaging_install import _load_plugin_entrypoint_module
    module = _load_plugin_entrypoint_module(f"lcm_surface_{assertions}_{adaptive}")
    monkeypatch.setenv("LCM_ASSERTIONS_ENABLED", str(assertions).lower())
    monkeypatch.setenv("LCM_ADAPTIVE_RETRIEVAL_ENABLED", str(adaptive).lower())
    monkeypatch.setattr(module, "_host_forwards_registered_tool_messages", lambda ctx: True)
    class Ctx:
        def __init__(self):
            self.tools = {}
        def register_context_engine(self, engine):
            self.engine = engine
        def register_tool(self, **kwargs):
            self.tools[kwargs["name"]] = kwargs
    ctx = Ctx()
    module.register(ctx)
    try:
        names = {schema["name"] for schema in ctx.engine.get_tool_schemas()}
        assert names == set(ctx.tools)
        assert ("lcm_query_state" in names) is assertions
        assert ("lcm_retrieve" in names) is adaptive
        assert len(names) == 13 + int(assertions) + int(adaptive)
        ctx.engine.on_session_start("surface", platform="cli")
        for name, enabled in (("lcm_query_state", assertions), ("lcm_retrieve", adaptive)):
            if not enabled:
                response = json.loads(ctx.engine.handle_tool_call(name, {}, messages=[{"role":"user", "content":"DO_NOT_INGEST_DISABLED"}]))
                assert response["status"] == "disabled"
                assert response["enable_with"].endswith("=true")
                assert ctx.engine._store.get_session_count("surface") == 0
    finally:
        ctx.engine.shutdown()


def test_fixed_operand_sum_remains_computed(engine):
    engine.ingest([
        {"role": "user", "content": "The first item cost $30."},
        {"role": "user", "content": "The second item cost $18."},
    ])
    rows = engine._store._conn.execute("SELECT store_id, content FROM messages ORDER BY store_id").fetchall()
    operands = [dict(store_id=row[0], span_start=0, span_end=len(row[1]), quote=row[1], value=value, unit="usd")
                for row, value in zip(rows, [30, 18])]
    result = json.loads(engine.handle_tool_call("lcm_compute", {
        "question": "What is the total of the two items?", "operands": operands,
    }))
    assert result["status"] == "computed"
    assert result["trace"]["result_value"] == 48


def test_adaptive_caller_slot_closure_cannot_certify_omitted_evidence(tmp_path):
    from .test_adaptive_retrieval import _engine, _append, _start, _search, _call, _operand
    instance = _engine(tmp_path)
    try:
        paris = _append(instance, "I visited Paris.")
        _append(instance, "I visited Rome.")
        started = _start(instance, question="How many cities did I visit?", operation="count_distinct", minimum_refs=1)
        evidence = _search(instance, started["retrieval_id"], paris)["evidence"][0]
        finished = _call(instance, action="finish", retrieval_id=started["retrieval_id"],
            resolved_slots=[{"slot_id":"visits", "evidence_refs":[evidence["citation"]]}],
            selected_refs=[evidence["citation"]], computation={"operands":[_operand(evidence, key="Paris")]})
        assert finished["status"] == "fallback"
        assert finished["computation"]["status"] == "subset_computed"
        assert finished["computation"]["trace"]["result_value"] == 1
        assert finished["query_view"]["persistence"]["status"] == "skipped"
    finally:
        instance.shutdown()

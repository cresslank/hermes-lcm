"""Closed source grammar over real SQLite messages; no assertion/model supplier."""
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from hermes_lcm.config import LCMConfig
from hermes_lcm.literal_record import validate_row
from hermes_lcm.store import MessageStore
from hermes_lcm.tools import lcm_evidence_pack, lcm_compile_evidence


def literal():
    return {"schema": "lcm.literal-record.v1", "entity": {"namespace": "inventory", "id": "A"},
        "predicate": {"namespace": "state", "id": "ready"},
        "time": {"basis": "atemporal", "start": None, "end": None},
        "scope": {"namespace": "site", "id": "S", "conditions": []},
        "quantity": {"kind": "non_quantity", "unit": None}, "value": True,
        "polarity": "positive", "modality": "possible"}


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "sources.db"))
    store = MessageStore(config.database_path, ingest_protection_config=config)
    yield SimpleNamespace(_config=config, _store=store, _assertions=None, _session_occurrence_dates={})
    store.close()


def row(engine, content, role="user"):
    sid = engine._store.append("source-session", {"role": role, "content": content})
    return engine._store.get(sid)


@pytest.mark.parametrize("role", ["user", "tool"])
def test_original_whole_record_bytes_and_immutable_coordinates(engine, role):
    text = " \n" + json.dumps(literal(), indent=2) + "\n "
    source = row(engine, text, role)
    value = validate_row(source, 0, len(text))
    assert value and value.source_bytes == text.encode()
    assert json.loads(dict(value.coordinate_pins)["time"]) == literal()["time"]
    assert json.loads(dict(value.source_attribution)["role"]) == role
    assert value.exact_ref == f"lcm:{source['store_id']}:0-{len(text)}"
    with pytest.raises(FrozenInstanceError):
        value.row_hash = "fake"


@pytest.mark.parametrize("field", ["entity", "predicate", "time", "scope", "quantity", "polarity", "modality", "value", "schema"])
@pytest.mark.parametrize("mutation", ["missing", "null", "unknown"])
def test_missing_and_unknown_coordinates_abstain(engine, field, mutation):
    data = literal()
    if mutation == "missing":
        del data[field]
    elif mutation == "null":
        data[field] = None
    elif field == "value":
        data[field] = {"model_guess": True}
    else:
        data[field] = {"unknown": True}
    text = json.dumps(data)
    assert validate_row(row(engine, text), 0, len(text)) is None


@pytest.mark.parametrize("bounds", [(-1, 10), (1, 10), (False, 10), (0, True), (0, 10)])
def test_partial_or_ambiguous_bounds_never_qualify(engine, bounds):
    text = json.dumps(literal())
    assert validate_row(row(engine, text), *bounds) is None


@pytest.mark.parametrize("time", [
    {"basis": "event_interval", "start": "2026-01-01", "end": "2026-01-02"},
    {"basis": "event_interval", "start": "2026-01-01T00:00:00-00:00", "end": "2026-02-01T00:00:00Z"},
    {"basis": "event_interval", "start": "2026-01-01T00:00:00+00:60", "end": "2026-02-01T00:00:00Z"},
    {"basis": "atemporal", "start": "2026-01-01T00:00:00Z", "end": None},
    {"basis": "observation_time", "start": None, "end": None},
])
def test_unsupported_time_semantics(engine, time):
    data = literal()
    data["time"] = time
    text = json.dumps(data)
    assert validate_row(row(engine, text), 0, len(text)) is None


@pytest.mark.parametrize("mode", ["pack", "proposal", "auto"])
def test_ordinary_tools_without_native_grant_do_not_mint_authority(engine, mode, monkeypatch):
    from hermes_lcm import tools
    monkeypatch.setattr(tools, "lcm_recall", lambda *a, **k: pytest.fail("extra retrieval"))
    text = json.dumps(literal())
    source = row(engine, text)
    ref = f"lcm:{source['store_id']}:0-{len(text)}"
    args = {"question": "What is the state?", "baseline_refs": [{"exact_ref": ref}], "budgets": {"max_retrieval_calls": 0}}
    if mode == "pack":
        result = lcm_evidence_pack(args, engine=engine)
    else:
        args["mode"] = mode
        args["proposal"] = {"version": "evidence-selector-v1", "selections": [{"claim_id": "c", "facet": "answer", "exact_ref": ref, "quote": text}], "missing_facets": []}
        result = lcm_compile_evidence(args, engine=engine)
    assert "source_propositions" not in json.loads(result)
    assert validate_row(source, 0, len(text)) is not None  # validation alone is NOT host authority


def test_assistant_reencoding_or_assertion_sidecar_never_qualifies(engine):
    text = json.dumps(literal())
    source = row(engine, text, "assistant")
    source.update(assertion_id="a" * 64, entity="A", validated=True, predicate="ready")
    assert validate_row(source, 0, len(text)) is None

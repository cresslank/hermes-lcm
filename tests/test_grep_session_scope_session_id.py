"""lcm_grep accepts a session id mistakenly passed as ``session_scope``."""

import json

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "grep_scope.db")))
    instance._session_id = "test-session"
    instance._store.append("test-session", {"role": "user", "content": "docker rollout current session"})
    instance._store.append("20260921_095506_1dec78d6", {"role": "user", "content": "docker rollout old session"})
    try:
        yield instance
    finally:
        instance.shutdown()


def _grep(engine, **args):
    return json.loads(engine.handle_tool_call("lcm_grep", {"query": "docker", "limit": 10, **args}))


def test_session_id_passed_as_scope_searches_that_session(engine):
    result = _grep(engine, session_scope="20260921_095506_1dec78d6")

    assert result["session_scope"] == "session"
    assert result["session_id"] == "20260921_095506_1dec78d6"
    assert [hit["session_id"] for hit in result["results"]] == ["20260921_095506_1dec78d6"]
    assert "ignored_session_scope" not in result
    assert "session_scope=session" in result["scope_note"]


def test_session_id_scope_with_matching_session_id_is_accepted(engine):
    result = _grep(
        engine,
        session_scope="20260921_095506_1dec78d6",
        session_id="20260921_095506_1dec78d6",
    )

    assert result["session_scope"] == "session"
    assert result["total_results"] == 1


def test_unknown_scope_that_is_not_a_stored_session_keeps_fallback(engine):
    result = _grep(engine, session_scope="20990101_000000_nothere")

    assert result["session_scope"] == "current"
    assert result["ignored_session_scope"] == "20990101_000000_nothere"
    assert [hit["session_id"] for hit in result["results"]] == ["test-session"]


def test_conflicting_session_id_is_not_overridden(engine):
    result = _grep(
        engine,
        session_scope="20260921_095506_1dec78d6",
        session_id="test-session",
    )

    # Conflicting arguments keep the documented behaviour: unknown scope with
    # a session_id falls back to current and reports the ignored scope.
    assert result.get("session_scope") != "session" or result.get("session_id") != "20260921_095506_1dec78d6"

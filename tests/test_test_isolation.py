"""Regression coverage for the suite-wide SQLite safety boundary."""

import sqlite3
import sys
from pathlib import Path

import pytest


conftest = next(
    module
    for module in tuple(sys.modules.values())
    if str(getattr(module, "__file__", "")).endswith("/tests/conftest.py")
)


def test_sqlite_guard_denies_real_home_forms_before_connect(monkeypatch, tmp_path):
    delegated = []

    def must_not_open(*args, **kwargs):
        delegated.append((args, kwargs))
        raise AssertionError("protected database reached sqlite3.connect")

    monkeypatch.setattr(conftest, "_ORIGINAL_SQLITE_CONNECT", must_not_open)

    live_db = Path("/Users/cresslank/.hermes/lcm.db")
    alias = tmp_path / "live-hermes-alias"
    alias.symlink_to(live_db.parent, target_is_directory=True)
    targets = (
        str(live_db),
        live_db.as_uri() + "?mode=ro",
        str(alias / live_db.name),
    )

    for target in targets:
        with pytest.raises(conftest.UnsafeSQLitePathError, match="protected Hermes home"):
            conftest._guarded_sqlite_connect(target, uri=target.startswith("file:"))

    assert delegated == []
    assert str(live_db) in conftest._SQLITE_DENIAL_AUDIT


def test_sqlite_guard_allows_temporary_paths_and_file_uris(tmp_path):
    for index, target in enumerate(
        (
            str(tmp_path / "direct.db"),
            (tmp_path / "uri.db").as_uri() + "?mode=rwc",
        )
    ):
        connection = sqlite3.connect(target, uri=target.startswith("file:"))
        try:
            connection.execute("CREATE TABLE proof (value INTEGER)")
            connection.execute("INSERT INTO proof VALUES (?)", (index,))
            assert connection.execute("SELECT value FROM proof").fetchone() == (index,)
        finally:
            connection.close()

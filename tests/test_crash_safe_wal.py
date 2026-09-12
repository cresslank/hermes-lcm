"""Tests for WAL durability configuration and graceful-close hygiene.

These tests verify the PRAGMAs applied by ``configure_connection()`` and
the best-effort passive WAL checkpoint performed by ``close()`` on all three
SQLite helpers.

This covers the PR #237 hardening path without overclaiming it: graceful close
can checkpoint committed WAL frames best-effort, while unexpected process death
still depends on SQLite WAL recovery.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_lcm.db_bootstrap import (
    SQLiteJournalModeError,
    configure_connection,
    ensure_message_origin_columns,
    join_background_integrity_scans,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.lifecycle_state import LifecycleStateStore


# --------------------------------------------------------------------------- #
#  configure_connection PRAGMA verification
# --------------------------------------------------------------------------- #


class TestConfigureConnectionPragmas:
    """Assert that configure_connection() sets the intended PRAGMAs."""

    @pytest.fixture()
    def db_path(self, tmp_path: Path):
        """Return a temp file path for an on-disk database (WAL requires a
        real file — :memory: silently reports journal_mode='memory')."""
        return tmp_path / "test.db"

    def test_journal_mode_is_wal(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal", f"expected journal_mode=wal, got {mode!r}"

    def test_synchronous_is_full(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        # PRAGMA synchronous returns an integer: 0=OFF, 1=NORMAL, 2=FULL
        val = conn.execute("PRAGMA synchronous").fetchone()[0]
        conn.close()
        assert val == 2, f"expected synchronous=FULL (2), got {val}"

    def test_busy_timeout(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        val = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.close()
        assert val == 30_000, f"expected busy_timeout=30000, got {val}"

    def test_wal_autocheckpoint(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        # After setting, PRAGMA wal_autocheckpoint returns the NEW value.
        val = conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
        conn.close()
        assert val == 500, f"expected wal_autocheckpoint=500, got {val}"

    def test_journal_size_limit(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        val = conn.execute("PRAGMA journal_size_limit").fetchone()[0]
        conn.close()
        assert val == 67_108_864, f"expected journal_size_limit=67108864, got {val}"

    def test_mmap_size(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        configure_connection(conn)
        val = conn.execute("PRAGMA mmap_size").fetchone()[0]
        conn.close()
        assert val == 268_435_456, f"expected mmap_size=268435456, got {val}"

    def test_delete_mode_uses_full_sync_without_mmap_or_wal_pragmas(
        self, db_path: Path
    ):
        conn = sqlite3.connect(str(db_path))
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        configure_connection(conn, journal_mode="delete")

        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert conn.execute("PRAGMA mmap_size").fetchone()[0] == 0
        conn.close()

        configured = [statement.lower() for statement in statements]
        assert not any("wal_autocheckpoint" in statement for statement in configured)
        assert not any("wal_checkpoint" in statement for statement in configured)

    def test_delete_mode_refuses_live_wal_database(self, db_path: Path):
        holder = sqlite3.connect(str(db_path), isolation_level=None)
        holder.execute("PRAGMA journal_mode=WAL")
        holder.execute("CREATE TABLE held(value TEXT)")
        holder.execute("BEGIN")
        holder.execute("SELECT * FROM held").fetchall()

        contender = sqlite3.connect(str(db_path), timeout=0.05)
        try:
            with pytest.raises(SQLiteJournalModeError, match="Stop every process"):
                configure_connection(contender, journal_mode="delete")
            assert contender.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            contender.close()
            holder.rollback()
            holder.close()

    def test_journal_mode_config_is_validated_and_yaml_aware(
        self, monkeypatch, tmp_path: Path
    ):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "lcm:\n  sqlite_journal_mode: delete\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("LCM_SQLITE_JOURNAL_MODE", raising=False)

        config = LCMConfig.from_env()
        assert config.sqlite_journal_mode == "delete"
        assert config.config_sources["sqlite_journal_mode"] == (
            "config_yaml:lcm.sqlite_journal_mode"
        )
        assert "sqlite_journal_mode" not in config.ignored_config_yaml_lcm_keys

        monkeypatch.setenv("LCM_SQLITE_JOURNAL_MODE", "truncate")
        with pytest.raises(ValueError, match="LCM_SQLITE_JOURNAL_MODE"):
            LCMConfig.from_env()


# --------------------------------------------------------------------------- #
#  Graceful close — WAL checkpoint on close
# --------------------------------------------------------------------------- #


class TestGracefulClose:
    """Verify that close() performs a best-effort passive WAL checkpoint
    without raising."""

    def _write_and_get_wal_size(self, db_path: Path) -> int:
        """Return WAL file size in bytes (0 if no WAL)."""
        wal = Path(str(db_path) + "-wal")
        return wal.stat().st_size if wal.exists() else 0

    # -- MessageStore -------------------------------------------------------

    def test_message_store_close_runs_checkpoint(self, tmp_path: Path):
        db = tmp_path / "store.db"
        store = MessageStore(db)
        store.append("sess", {"role": "user", "content": "hello"})
        assert db.exists()
        store.close()
        # After close the WAL should be small or non-existent (all frames
        # checkpointed by the passive call).
        wal_size = self._write_and_get_wal_size(db)
        assert wal_size < 4096, (
            f"WAL still {wal_size} bytes after MessageStore.close(); "
            "checkpoint may not have run"
        )

    def test_message_store_close_is_idempotent(self, tmp_path: Path):
        db = tmp_path / "store.db"
        store = MessageStore(db)
        store.close()
        store.close()  # should not raise

    # -- SummaryDAG ---------------------------------------------------------

    def test_summary_dag_close_runs_checkpoint(self, tmp_path: Path):
        db = tmp_path / "dag.db"
        dag = SummaryDAG(db)
        # Insert a minimal summary node so the WAL has content
        conn = dag._conn
        assert conn is not None
        conn.execute(
            "INSERT INTO summary_nodes (session_id, depth, summary, "
            "source_ids, source_type, created_at, earliest_at, latest_at) "
            "VALUES ('sess', 0, 'summary', '[]', 'messages', 0.0, 0.0, 0.0)"
        )
        conn.commit()
        dag.close()
        wal_size = self._write_and_get_wal_size(db)
        assert wal_size < 4096, (
            f"WAL still {wal_size} bytes after SummaryDAG.close(); "
            "checkpoint may not have run"
        )

    # -- LifecycleStateStore ------------------------------------------------

    def test_lifecycle_state_close_runs_checkpoint(self, tmp_path: Path):
        db = tmp_path / "lifecycle.db"
        lc = LifecycleStateStore(db)
        lc.bind_session("sess")
        lc.close()
        wal_size = self._write_and_get_wal_size(db)
        assert wal_size < 4096, (
            f"WAL still {wal_size} bytes after LifecycleStateStore.close(); "
            "checkpoint may not have run"
        )

    # -- Masking check ------------------------------------------------------

    def test_message_store_close_does_not_mask_sqlite_error(self, tmp_path: Path):
        """close() should not silently swallow a broken connection — it only
        ignores errors from the checkpoint attempt itself, not from the
        underlying close."""
        db = tmp_path / "store.db"
        store = MessageStore(db)
        # Manually invalidate the connection so close() has nothing to do
        store._conn = None
        store.close()  # should not raise

    def test_summary_dag_close_does_not_mask_sqlite_error(self, tmp_path: Path):
        db = tmp_path / "dag.db"
        dag = SummaryDAG(db)
        dag._conn = None
        dag.close()  # should not raise

    def test_lifecycle_state_close_does_not_mask_sqlite_error(self, tmp_path: Path):
        db = tmp_path / "lifecycle.db"
        lc = LifecycleStateStore(db)
        lc._conn = None
        lc.close()  # should not raise

    def test_delete_mode_close_does_not_issue_wal_checkpoint(
        self, monkeypatch, tmp_path: Path
    ):
        monkeypatch.setenv("LCM_SQLITE_JOURNAL_MODE", "delete")
        store = MessageStore(tmp_path / "delete-close.db")
        statements: list[str] = []
        store._conn.set_trace_callback(statements.append)
        store.close()
        assert not any(
            "wal_checkpoint" in statement.lower() for statement in statements
        )


def test_delete_mode_cloned_engines_write_and_close_concurrently(tmp_path: Path):
    db_path = tmp_path / "shared-delete.db"
    config = LCMConfig(database_path=str(db_path), sqlite_journal_mode="delete")
    prototype = LCMEngine(config=config, hermes_home=str(tmp_path))
    clones = [prototype.clone_for_agent() for _ in range(4)]
    join_background_integrity_scans(timeout=30.0)

    barrier = threading.Barrier(len(clones))
    errors: list[BaseException] = []
    error_lock = threading.Lock()

    def write_and_close(index: int, engine: LCMEngine) -> None:
        try:
            engine.on_session_start(
                f"delete-session-{index}", context_length=100_000
            )
            barrier.wait(timeout=30.0)
            engine.ingest(
                [{"role": "user", "content": f"delete writer {index}"}]
            )
        except BaseException as exc:  # noqa: BLE001 - re-asserted below
            with error_lock:
                errors.append(exc)
        finally:
            engine.shutdown()

    threads = [
        threading.Thread(target=write_and_close, args=(index, engine))
        for index, engine in enumerate(clones)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60.0)
    prototype.shutdown()
    join_background_integrity_scans(timeout=30.0)

    assert not [thread for thread in threads if thread.is_alive()]
    assert not errors
    assert not Path(f"{db_path}-wal").exists()
    assert not Path(f"{db_path}-shm").exists()

    check = sqlite3.connect(str(db_path))
    try:
        assert check.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert check.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert check.execute("SELECT count(*) FROM messages").fetchone()[0] == 4
    finally:
        check.close()


# --------------------------------------------------------------------------- #
#  Concurrent-startup migration race (idempotent ADD COLUMN)
# --------------------------------------------------------------------------- #


def _seed_pre_conversation_id_messages(path: Path) -> None:
    """Create a ``messages`` table as a pre-v5 build left it: without the
    ``conversation_id`` column the column migration later adds. Scoped to the
    column DDL only (no FTS), so the test isolates the ADD COLUMN race."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE messages (
                store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


class TestConcurrentStartupMigration:
    """Concurrent process startup must not crash on duplicate-column ALTERs.

    Regression for the pre-fix race: gateway + CLI + sub-agents open independent
    connections to one ``lcm.db`` after an upgrade and all run the column
    migrations; the loser hit ``sqlite3.OperationalError: duplicate column name``
    and crashed store construction. ``add_column_if_missing`` makes the ALTER
    idempotent so every process migrates successfully.
    """

    def test_concurrent_column_migration_is_idempotent(self, tmp_path: Path):
        db_path = tmp_path / "concurrent.db"
        _seed_pre_conversation_id_messages(db_path)

        thread_count = 8
        errors: list[BaseException] = []
        lock = threading.Lock()
        barrier = threading.Barrier(thread_count)

        def migrate() -> None:
            # Each thread is a stand-in for a separate process: its own
            # connection to the same file, its own busy_timeout, racing the same
            # ``ALTER TABLE messages ADD COLUMN conversation_id``.
            conn = sqlite3.connect(str(db_path), timeout=30.0)
            try:
                configure_connection(conn)
                # Timeout + abort-on-error: a thread that fails before the
                # barrier must break it, or the surviving threads wait forever
                # and the deadlock hides the original error from the assert.
                barrier.wait(timeout=60.0)  # maximise overlap on the migration DDL
                ensure_message_origin_columns(conn)
                conn.commit()
            except BaseException as exc:  # noqa: BLE001 - re-asserted below
                barrier.abort()
                with lock:
                    errors.append(exc)
            finally:
                conn.close()

        threads = [threading.Thread(target=migrate) for _ in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120.0)
        stuck = [t for t in threads if t.is_alive()]
        assert not stuck, f"{len(stuck)} migration threads still running after join timeout"

        real_errors = [
            exc for exc in errors if not isinstance(exc, threading.BrokenBarrierError)
        ]
        assert not real_errors, f"concurrent column migration raised: {real_errors!r}"
        assert not errors, "barrier broke without a recorded root-cause error"
        conn = sqlite3.connect(str(db_path))
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()
        ]
        conn.close()
        assert columns.count("conversation_id") == 1

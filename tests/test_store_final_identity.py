"""Final point reads stay bound to the opened store, not a reusable filename."""

import sqlite3
import threading

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.store import MessageStore


@pytest.mark.parametrize("journal_mode", ["DELETE", "WAL"])
@pytest.mark.parametrize("state", ["ordinary", "deleted", "replacement", "transaction", "busy", "missing", "closed", "memory"])
def test_final_read_preserves_store_identity_and_connection_state(tmp_path, journal_mode, state):
    path = tmp_path / "lcm.db"
    config = LCMConfig(database_path=str(path), sqlite_journal_mode=journal_mode)
    store = MessageStore(":memory:" if state == "memory" else path, ingest_protection_config=config)
    blocker = None
    moved = []
    try:
        sid = store.append("session-A", {"role": "user", "content": "Selected original source"})
        expected = store.get(sid)
        if state == "replacement":
            backup = sqlite3.connect(tmp_path / "backup.db")
            try:
                store.backup(backup)
            finally:
                backup.close()
        if state in {"deleted", "replacement"}:
            assert store.delete_session_messages("session-A") == 1
            assert store.get(sid) is None
        if state in {"replacement", "missing"}:
            for suffix in ("", "-wal", "-shm"):
                source = path.with_name(path.name + suffix)
                if source.exists():
                    detached = path.with_name("detached-" + source.name)
                    source.rename(detached)
                    moved.append((source, detached))
            if state == "replacement":
                (tmp_path / "backup.db").rename(path)
                assert path.stat().st_ino != moved[0][1].stat().st_ino
                assert store.get(sid) is None
        if state == "closed":
            store.close()
        else:
            store.connection.execute("PRAGMA busy_timeout=1379")
            if state == "transaction":
                store.connection.execute("BEGIN")
                store.connection.execute("UPDATE messages SET content='uncommitted' WHERE store_id=?", (sid,))
            if state == "busy":
                blocker = sqlite3.connect(path, timeout=0)
                blocker.execute("BEGIN EXCLUSIVE")
            before_transaction = store.connection.in_transaction

        # A competing thread must finish even while the ordinary write lock is
        # held. Neither busy timeout nor an existing transaction may be changed.
        results, errors = [], []
        def read():
            try:
                results.append(store._get_for_local_final(sid))
            except BaseException as exc:
                errors.append(exc)
        with store._write_lock:
            worker = threading.Thread(target=read)
            worker.start()
            worker.join(3)
        assert not worker.is_alive(), "final read waited for the store writer lock"
        if state != "closed":
            assert store.connection.in_transaction == before_transaction
            assert store.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1379
        if state == "busy" and journal_mode == "DELETE":
            assert len(errors) == 1 and isinstance(errors[0], sqlite3.OperationalError)
            assert "locked" in str(errors[0])
        else:
            assert not errors
            assert results == ([expected] if state in {"ordinary", "transaction", "busy"} else [None])
        if state == "replacement":
            # A genuinely new store may bind to the replacement; the old store
            # may not refresh its authority by following that same pathname.
            fresh = MessageStore(path, ingest_protection_config=config)
            try:
                assert fresh._get_for_local_final(sid) == expected
                assert store._get_for_local_final(sid) is None
            finally:
                fresh.close()
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        if moved:
            for suffix in ("", "-wal", "-shm"):
                candidate = path.with_name(path.name + suffix)
                if candidate.exists():
                    candidate.unlink()
            for source, detached in moved:
                detached.rename(source)
        if store.connection is not None:
            store.connection.rollback()
        store.close()

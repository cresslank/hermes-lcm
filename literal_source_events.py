"""Native append/deletion outbox for enrolled literal originals, not a watcher.

Compact metadata shares the source owner's existing transaction. Delivery hints
are released only after commit. A restart can inspect these facts but cannot use
an old process's custody as fresh review or send permission.
"""
import hashlib
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

from .literal_record import validate_row

PREFIX = 'literal-correction-event:'


def prepare(store, ids=(), *, unavailable_session=None):
    """Optional metadata cannot strand an otherwise valid native source write."""
    if getattr(store, '_literal_correction_provider', None) is None:
        return ()
    # A deletion calls us before its first DML statement. Keep the savepoint
    # inside the source owner's transaction; RELEASE must not commit the outbox.
    if not store._conn.in_transaction:
        store._conn.execute("BEGIN")
    store._conn.execute("SAVEPOINT literal_correction_events")
    try:
        events = _prepare(store, ids, unavailable_session=unavailable_session)
    except Exception:
        store._conn.execute("ROLLBACK TO literal_correction_events")
        store._conn.execute("RELEASE literal_correction_events")
        logger.debug("optional correction metadata unavailable", exc_info=True)
        return ()
    store._conn.execute("RELEASE literal_correction_events")
    return events


def _prepare(store, ids=(), *, unavailable_session=None):
    provider = getattr(store, '_literal_correction_provider', None)
    if provider is None:
        return ()
    with provider.lock:
        enrolled = tuple(c for c in provider.correction_custody.values()
            if c.engine._store.db_path == store.db_path
            and provider.literal_source_binding(c.engine) == c.binding)
    if not enrolled:
        return ()
    from .store import _MESSAGE_SELECT_COLUMNS
    from agent.supervision_claim_uses import comparable
    events = []
    if len(ids) > 8:
        return ()  # explicit unsupported batch, never a silently truncated pair set
    for sid in ids:
        row = store._conn.execute(f'SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages WHERE store_id=?', (sid,)).fetchone()
        row = store._row_to_dict(row) if row else None
        text = row.get('content') if row else None
        new = validate_row(row, 0, len(text)) if type(text) is str else None
        if new is None:
            continue
        matches = [c for c in enrolled if row['session_id'] == c.engine.current_session_id and comparable(c.record, new)]
        if len(matches) > 4:
            return ()
        for custody in matches:
            a, b = dict(custody.record.coordinate_pins), dict(new.coordinate_pins)
            # Same native coordinates, changed proposition: numeric conflicts
            # are code-owned; other relations require separately granted judgment.
            if any(a[k] != b[k] for k in ('value', 'polarity', 'modality')):
                events.append((custody, new, 'source_changed'))
    if unavailable_session is not None:
        affected = [c for c in enrolled if c.engine.current_session_id == unavailable_session]
        if len(affected) > 4:
            return ()
        events.extend((c, None, 'source_unavailable') for c in affected)
    pending = []
    if store._conn.execute('SELECT count(*) FROM metadata WHERE key LIKE ?', (PREFIX+'%',)).fetchone()[0] + len(events) > 64:
        return ()
    for custody, new, kind in events:
        original_id = custody.original.row['id']
        body = dict(original_id=original_id, old_ref=custody.record.exact_ref,
            old_hash=custody.record.row_hash, new_ref=new.exact_ref if new else None,
            new_hash=new.row_hash if new else None, kind=kind,
            session_id=custody.engine.current_session_id)
        raw = json.dumps(body, sort_keys=True, separators=(',', ':'))
        ident = PREFIX + hashlib.sha256(raw.encode()).hexdigest()
        changed = store._conn.execute('INSERT OR IGNORE INTO metadata(key,value) VALUES(?,?)', (ident, raw)).rowcount
        if changed:
            pending.append((provider, custody, new, kind, ident, raw))
    return tuple(pending)


def committed(events):
    # The source is already committed. Callback uncertainty must not turn that
    # completed read/write into a retryable failure or replay an external effect.
    for event in events:
        try:
            _committed((event,))
        except Exception:
            logger.debug("optional postcommit correction ingress unavailable", exc_info=True)


def _committed(events):
    if not events:
        return
    from agent.supervision_corrections import source_event_received
    for provider, custody, new, kind, ident, raw in events:
        with provider.lock:
            if (provider.literal_source_binding(custody.engine) != custody.binding
                    or len(provider.correction_candidates) >= 64):
                continue
            if kind == "source_unavailable":
                custody.unavailable = True
            key = (ident, new.exact_ref if new else custody.record.exact_ref,
                new.row_hash if new else custody.record.row_hash, custody.original.row['id'])
            provider.correction_candidates[key] = (custody.engine, custody.binding, new)
            provider.correction_events[ident] = (key, raw, kind)
        source_event_received(custody.original.rt, provider, custody.original.row['id'], key, kind)


def event_current(provider, ident):
    entry = provider.correction_events.get(ident)
    if entry is None:
        return False
    candidate = provider.correction_candidates.get(entry[0])
    if candidate is None:
        return False
    engine, binding, _ = candidate
    if provider.literal_source_binding(engine) != binding:
        return False
    store = engine._store
    if store._local_final_file_identity() != store._local_final_identity:
        return False
    try:
        conn = sqlite3.connect(store.db_path.resolve().as_uri()+'?mode=ro', uri=True, timeout=0)
        try:
            row = conn.execute('SELECT value FROM metadata WHERE key=?', (ident,)).fetchone()
        finally:
            conn.close()
        return row == (entry[1],) and store._local_final_file_identity() == store._local_final_identity
    except (sqlite3.Error, OSError):
        return False

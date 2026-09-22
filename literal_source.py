"""Optional native publication of whole records observed by ordinary LCM tools.

No additional retrieval or inference. Revalidation reads only already selected
row IDs; it cannot search, follow refs, consult assertions, or broaden scope.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import json
import sqlite3
import threading
import uuid
import weakref

from .literal_record import validate_row

_active: ContextVar = ContextVar("lcm_literal_source_invocation", default=None)
VERSION = "supervision.literal-sources.v1"


@dataclass
class _Capture:
    engine: object
    provider: object
    invocation: object
    refs: dict
    overflow: bool = False


def install_literal_source_owner(engine, facade):
    """Called only after the ordinary context-engine registration succeeds."""
    if facade is None:
        return None
    try:
        from agent.supervision_literal_sources import LiteralSourceProviderV1, LiteralSourceRecordV1
    except ImportError:
        return None

    class Provider(LiteralSourceProviderV1):
        def __init__(self):
            self.engines = weakref.WeakKeyDictionary({engine: (engine._store, engine._hermes_home, engine._config.database_path)})
            self.records = {}
            self.invocations = set()
            self.lock = threading.RLock()

        def owns_engine(self, candidate):
            try:
                return self.engines.get(candidate) == (candidate._store, candidate._hermes_home, candidate._config.database_path)
            except (AttributeError, TypeError):
                return False

        def bind_clone(self, parent, clone):
            if (parent in self.engines and type(clone) is type(engine)
                    and clone._hermes_home == engine._hermes_home
                    and clone._config.database_path == engine._config.database_path):
                self.engines[clone] = (clone._store, clone._hermes_home, clone._config.database_path)
                clone._literal_source_provider = self

        def capture(self, capture, row, start, end):
            record = validate_row(row, start, end)
            if record is None:
                return
            key = (record.exact_ref, record.row_hash)
            if key in capture.refs:
                return
            with self.lock:
                if capture.invocation.id not in self.invocations:
                    capture.overflow = True
                    return
                if len(capture.refs) >= 8 or len(self.records) >= 64:
                    capture.overflow = True
                    return
                ref = uuid.uuid4().hex
                self.records[(capture.invocation.id, ref)] = (
                    capture.engine, record,
                    (capture.engine.current_session_id, capture.engine.current_conversation_id))
                capture.refs[key] = ref

        def resolve_literal_source(self, candidate, invocation_id, source_ref):
            with self.lock:
                stored = self.records.get((invocation_id, source_ref))
                if (stored is None or stored[0] is not candidate or not self.owns_engine(candidate)
                        or stored[2] != (candidate.current_session_id, candidate.current_conversation_id)):
                    return None
                original = stored[1]
                store_id = int(original.exact_ref.split(":")[1])
                # Point read of the original hydrated row, not evidence retrieval.
                try:
                    row = candidate._store.get(store_id)
                except (sqlite3.Error, OSError):
                    return None
                current = validate_row(row, *original.record_span) if row is not None else None
                if (current != original or not self.owns_engine(candidate)
                        or stored[2] != (candidate.current_session_id, candidate.current_conversation_id)):
                    self.records.pop((invocation_id, source_ref), None)
                    return None
                return LiteralSourceRecordV1(**vars(original))

        def release_literal_sources(self, invocation_id):
            with self.lock:
                self.invocations.discard(invocation_id)
                for key in tuple(self.records):
                    if key[0] == invocation_id:
                        del self.records[key]

    provider = Provider()
    register = getattr(facade, "register_literal_source_owner", None)
    if register is None:
        return None
    registration = register(version=VERSION, engine=engine, provider=provider)
    if registration is not None:
        engine._literal_source_provider = provider
    return registration


def observe_hydrated(engine, row, start, end):
    capture = _active.get()
    if capture is not None and capture.engine is engine and not capture.overflow:
        capture.provider.capture(capture, row, start, end)


def literal_source_tool(function):
    """Capture only within an actual pack/compile invocation, including auto mode."""
    @wraps(function)
    def wrapped(args, **kwargs):
        from .tools import _require_engine
        engine = _require_engine(kwargs)
        facade = getattr(engine, "supervision", None)
        provider = getattr(engine, "_literal_source_provider", None)
        begin = getattr(facade, "begin_literal_sources", None)
        invocation = begin(version=VERSION, engine=engine) if provider is not None and begin is not None else None
        if invocation is None or provider is None or facade is None:
            return function(args, **kwargs)
        with provider.lock:
            provider.invocations.add(invocation.id)
        capture = _Capture(engine, provider, invocation, {})
        token = _active.set(capture)
        published = ()
        try:
            result = function(args, **kwargs)
            refs = facade.publish_literal_sources(version=VERSION, invocation=invocation,
                source_refs=tuple(capture.refs.values()) if not capture.overflow else ())
            published = refs
            if refs:
                payload = json.loads(result)
                payload["source_propositions"] = list(refs)
                return json.dumps(payload, ensure_ascii=False)
            return result
        finally:
            _active.reset(token)
            # Idempotent cancellation of a still-pending invocation, not release
            # of successfully published refs retained by the source owner.
            facade.cancel_literal_sources(version=VERSION, invocation=invocation)
            if not published:
                provider.release_literal_sources(invocation.id)
    return wrapped

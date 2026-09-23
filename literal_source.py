"""Optional native publication of whole records observed by ordinary LCM tools.

No additional retrieval or inference. Revalidation reads only already selected
row IDs; it cannot search, follow refs, consult assertions, or broaden scope.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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
class _Binding:
    generation: object = field(default_factory=object)
    transitions: int = 0
    ready: bool = True


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
    if not hasattr(LiteralSourceProviderV1, "literal_source_binding_fence"):
        # Older v1 hosts cannot fence lifecycle generations; keep ordinary tools.
        return None

    from .literal_source_final import FinalSourceOwner

    from .literal_source_correction import CorrectionSourceOwner

    class Provider(CorrectionSourceOwner, FinalSourceOwner, LiteralSourceProviderV1):
        def __init__(self):
            self.engines = weakref.WeakKeyDictionary({engine: (engine._store, engine._hermes_home, engine._config.database_path)})
            self.bindings = weakref.WeakKeyDictionary({engine: _Binding()})
            self.records = {}
            self.final_selections = set()
            self.final_permissions = set()
            self.correction_custody = {}
            self.correction_candidates = {}
            self.correction_events = {}
            engine._store._literal_correction_provider = self
            self.invocations = set()
            self.lock = threading.RLock()

        def owns_engine(self, candidate):
            try:
                return self.engines.get(candidate) == (candidate._store, candidate._hermes_home, candidate._config.database_path)
            except (AttributeError, TypeError):
                return False

        def literal_source_binding(self, candidate):
            with self.lock:
                state = self.bindings.get(candidate)
                if (state is None or not state.ready or state.transitions
                        or not self.owns_engine(candidate)):
                    return None
                return (state.generation, candidate.current_session_id, candidate.current_conversation_id)

        @contextmanager
        def literal_source_binding_fence(self, candidate, binding):
            # Final acceptance and lifecycle invalidation share this short lock.
            # No row reads, host callbacks, or host graph locks beneath it.
            with self.lock:
                yield binding is not None and self.literal_source_binding(candidate) == binding

        @contextmanager
        def lifecycle(self, candidate, *, resume):
            with self.lock:
                state = self.bindings[candidate]
                state.generation = object()  # irreversible even for A -> B -> A
                self.invalidate_final_sources(candidate)
                state.transitions += 1
                state.ready = False
            succeeded = False
            try:
                yield
                succeeded = True
            finally:
                with self.lock:
                    state.transitions -= 1
                    if not state.transitions:
                        state.ready = succeeded and resume

        def bind_clone(self, parent, clone):
            if (parent in self.engines and type(clone) is type(engine)
                    and clone._hermes_home == engine._hermes_home
                    and clone._config.database_path == engine._config.database_path):
                self.engines[clone] = (clone._store, clone._hermes_home, clone._config.database_path)
                self.bindings[clone] = _Binding()
                clone._literal_source_provider = self
                clone._store._literal_correction_provider = self

        def capture(self, capture, row, start, end):
            record = validate_row(row, start, end)
            if record is None:
                return
            key = (record.exact_ref, record.row_hash)
            if key in capture.refs:
                return
            with self.lock:
                if (capture.invocation.id not in self.invocations
                        or self.literal_source_binding(capture.engine) != capture.invocation.binding):
                    capture.overflow = True
                    return
                if len(capture.refs) >= 8 or self.final_capacity() >= 64:
                    capture.overflow = True
                    return
                ref = uuid.uuid4().hex
                self.records[(capture.invocation.id, ref)] = (
                    capture.engine, record,
                    capture.invocation.binding)
                capture.refs[key] = ref

        def resolve_literal_source(self, candidate, invocation_id, source_ref):
            with self.lock:
                stored = self.records.get((invocation_id, source_ref))
                if (stored is None or stored[0] is not candidate or not self.owns_engine(candidate)
                        or stored[2] != self.literal_source_binding(candidate)):
                    return None
                original = stored[1]
                store_id = int(original.exact_ref.split(":")[1])
            # Source I/O must not hold the lifecycle fence: the host takes that
            # fence under its graph locks only after this resolver returns.
            try:
                row = candidate._store.get(store_id)
            except (sqlite3.Error, OSError):
                return None
            current = validate_row(row, *original.record_span) if row is not None else None
            with self.lock:
                if (self.records.get((invocation_id, source_ref)) is not stored
                        or current != original or not self.owns_engine(candidate)
                        or stored[2] != self.literal_source_binding(candidate)):
                    self.records.pop((invocation_id, source_ref), None)
                    return None
                return LiteralSourceRecordV1(**vars(original))

        def working_literal_source(self, candidate, exact_ref):
            # This purpose is request-bound, not another publication lifetime.
            from .decision_adapter import _EXACT
            match = _EXACT.fullmatch(exact_ref) if type(exact_ref) is str else None
            binding = self.literal_source_binding(candidate)
            if match is None or binding is None:
                return None
            sid, start, end = map(int, match.groups())
            try:
                row = candidate._store.get_for_working_premise(sid)
            except (sqlite3.Error, OSError):
                return None
            if row is None or row.get("session_id") != candidate.current_session_id:
                return None
            record = validate_row(row, start, end)
            if record is None or self.literal_source_binding(candidate) != binding:
                return None
            return LiteralSourceRecordV1(**vars(record))

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


def literal_source_lifecycle(function):
    """Invalidate before mutation; keep publication unavailable during rebinding.

    Only a successful session start reopens capture. The native invocation's
    generation never survives a lifecycle boundary, even if IDs return to A.
    """
    @wraps(function)
    def wrapped(engine, *args, **kwargs):
        provider = getattr(engine, "_literal_source_provider", None)
        if provider is None:
            return function(engine, *args, **kwargs)
        with provider.lifecycle(engine, resume=function.__name__ == "on_session_start"):
            return function(engine, *args, **kwargs)
    return wrapped


def observe_hydrated(engine, row, start, end):
    provider = getattr(engine, "_literal_source_provider", None)
    if provider is not None:
        provider.capture_correction_candidate(engine, row, start, end)
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

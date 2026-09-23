"""Ephemeral, native-local final-use authority of the registered LCM row owner.

No publication resurrection, restart recovery, row-ID API, or egress permission.
The source selection and one-shot final permission are deliberately different
objects. Source validation uses the original validator and a zero-wait read of
only the selected whole row. Nothing is persisted here.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import sqlite3

from .literal_record import validate_row


@dataclass(eq=False, frozen=True)
class _Selection:
    engine: object
    record: object
    binding: object
    scope: tuple
    consumer: object


@dataclass(eq=False, frozen=True)
class _Permission:
    selection: _Selection
    phase: object


class FinalSourceOwner:
    """Mixed into the existing registered literal owner, sharing its lifecycle lock."""
    def final_capacity(self):
        return len(self.records) + len(self.final_selections) + len(self.final_permissions)

    def capture_final_source(self, engine, invocation_id, source_ref, scope, consumer):
        from agent.supervision_final_use import NativeFinalUse, selection_scope
        with self.lock:
            row = self.records.get((invocation_id, source_ref))
            rt = consumer.source.facade._active_runtime() if type(consumer) is NativeFinalUse else None
            if (row is None or row[0] is not engine or rt is None
                    or consumer.source.provider is not self or not consumer.current(rt)
                    or type(scope) is not tuple or len(scope) != 3
                    or scope[:2] != selection_scope(rt) or type(scope[2]) is not str or len(scope[2]) != 64
                    or self.final_capacity() >= 64
                    or row[2] != self.literal_source_binding(engine)):
                return None
            selected = _Selection(engine, row[1], row[2], scope, consumer)
            self.final_selections.add(selected)
            return selected

    def authorize_final_source(self, engine, selection, scope, consumer, phase):
        from agent.supervision_final_use import _FinalPhase
        with self.lock:
            if selection not in self.final_selections:
                return None
            self.final_selections.remove(selection)  # one attempt, including unavailable I/O
            if (type(phase) is not _FinalPhase or selection.engine is not engine
                    or selection.consumer is not consumer or selection.scope != scope
                    or phase.engine is not engine or phase.selection_scope != scope
                    or not phase.current(consumer)
                    or selection.binding != self.literal_source_binding(engine)):
                return None
            original = selection.record
        # Do not use the old publication resolver. Its lifetime remains unchanged.
        # This read cannot create a database or wait for a SQLite writer.
        try:
            row = engine._store._get_for_local_final(int(original.exact_ref.split(":")[1]))
        except (sqlite3.Error, OSError):
            return None
        current = validate_row(row, *original.record_span) if row is not None else None
        with self.lock:
            if (current != original or not phase.current(consumer)
                    or selection.binding != self.literal_source_binding(engine)):
                return None
            permission = _Permission(selection, phase)
            self.final_permissions.add(permission)
            return permission

    def final_source_current(self, permission):
        if permission not in self.final_permissions:
            return False
        s = permission.selection
        return (s.binding == self.literal_source_binding(s.engine)
                and permission.phase.current(s.consumer))

    @contextmanager
    def final_source_fence(self, permission):
        with self.lock:
            yield self.final_source_current(permission)

    def release_final_source(self, token):
        with self.lock:
            self.final_selections.discard(token)
            self.final_permissions.discard(token)

    def invalidate_final_sources(self, candidate):
        self.final_selections.difference_update(
            s for s in tuple(self.final_selections) if s.engine is candidate)
        self.final_permissions.difference_update(
            p for p in tuple(self.final_permissions) if p.selection.engine is candidate)

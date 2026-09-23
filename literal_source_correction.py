"""Bounded historical custody and fresh local correction reads.

Enrollment is provenance, not a retained publication grant. Only the native
original-output owner can enroll; only the current main correction owner can
consume. Restart intentionally restores no permissions.
"""
from dataclasses import dataclass
import sqlite3

from .literal_record import validate_row


@dataclass(eq=False)
class Custody:
    engine: object
    binding: object
    original: object
    record: object
    registration: object
    unavailable: bool = False


class CorrectionSourceOwner:
    def release_correction_sources(self):
        with self.lock:
            self.correction_custody.clear()
            self.correction_candidates.clear()
            self.correction_events.clear()

    def correction_event_current(self, ident):
        from .literal_source_events import event_current
        return event_current(self, ident)

    def enroll_correction_source(self, attempt):
        from agent.supervision_original_output import _Attempt
        from agent.supervision_corrections import enabled
        if type(attempt) is not _Attempt or not enabled(attempt.rt):
            return
        source = attempt.rt.agent().context_compressor
        with self.lock:
            if (not self.owns_engine(source) or not attempt.current()
                    or attempt.row['status'] != 'original_emitted'
                    or len(self.correction_custody) >= 64):
                return
            # This record came from the selected native final-use provenance,
            # never a model ref or bytes copied out of a tool response.
            record = attempt.source_record
            binding = self.literal_source_binding(source)
            if binding is None:
                return
            from agent.supervision_corrections import source_for
            self.correction_custody[attempt.row['id']] = Custody(
                source, binding, attempt, record, source_for(attempt.rt))

    def correction_pair(self, owner, original_id, candidate):
        from agent.supervision_corrections import Review
        if type(owner) is not Review or not owner.current():
            return None
        with self.lock:
            custody = self.correction_custody.get(original_id)
            if (custody is None or custody.unavailable or not self.correction_current(custody, owner)
                    or candidate not in self.correction_candidates):
                return None
            engine = custody.engine
            new = self.correction_candidates[candidate]
            if new[0] is not engine or new[1] != custody.binding:
                return None
            if new[2] is None:
                return None
            originals = (custody.record, new[2])
        checked = []
        for original in originals:
            try:
                row = engine._store._get_for_local_final(int(original.exact_ref.split(':')[1]))
            except (OSError, sqlite3.Error):
                return None
            current = validate_row(row, *original.record_span) if row is not None else None
            if current is None or vars(current) != vars(original) or row.get('session_id') != engine.current_session_id:
                return None
            checked.append(current)
        with self.lock:
            if self.correction_current(custody, owner) and self.correction_candidates.get(candidate) is new:
                return tuple(checked)
        return None

    def correction_current(self, custody, owner):
        from agent.supervision_corrections import source_for
        return (owner.current() and source_for(owner.rt) is custody.registration
            and custody.engine is owner.rt.agent().context_compressor
            and self.literal_source_binding(custody.engine) == custody.binding
            and self.correction_custody.get(custody.original.row['id']) is custody)

    def capture_correction_candidate(self, engine, row, start, end):
        """Actual native hydration is the producer, independent of tool JSON."""
        from agent.subagent_lifecycle import get_active_subagent_parent
        from agent.supervision_claim_uses import comparable
        from agent.supervision_corrections import candidate_received
        agent = get_active_subagent_parent()
        child = getattr(agent, '_subagent_id', None)
        record = validate_row(row, start, end)
        if record is None or not child:
            return
        with self.lock:
            matches = [c for c in self.correction_custody.values()
                if c.engine._store.db_path == engine._store.db_path
                and row.get('session_id') == c.engine.current_session_id
                and comparable(c.record, record)
                and self.literal_source_binding(c.engine) == c.binding]
            if not 1 <= len(matches) <= 4 or len(self.correction_candidates) >= 64:
                return
            for c in matches:
                key = (child, record.exact_ref, record.row_hash, c.original.row['id'])
                self.correction_candidates.setdefault(key, (c.engine, c.binding, record))
        # Host admission later verifies the committed native launch/child binding.
        # No source/graph lock crosses this call and no worker may send a notice.
        for c in matches:
            candidate_received(c.original.rt, self, c.original.row['id'],
                (child, record.exact_ref, record.row_hash, c.original.row['id']))

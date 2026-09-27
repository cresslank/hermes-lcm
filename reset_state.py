"""Session-scoped runtime-state resets.

Extracted verbatim from :mod:`hermes_lcm.engine` as ``ResetStateMixin``
(WS5 seam). The methods clear the session-scoped counters, compaction
progress, and per-turn placeholder-boundary bookkeeping when a session is
reset or rolled over. State stays on the engine (accessed via ``self``);
mixing this in leaves every call site and ``self._*`` reference unchanged.
"""

import hashlib
import json
import uuid


class ResetStateMixin:
    def _reset_session_counters(self) -> None:
        """Reset session-scoped counters and token tracking.

        Safe to call on boundary skip because it does not affect compaction progress.
        """
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_reasoning_tokens = 0
        self.cache_metrics_available = False
        self._compaction_telemetry_counter_epoch = uuid.uuid4().hex
        self._context_probed = False
        self._context_probe_persistable = False
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""
        self._last_boundary_skip_time = 0
        self._compaction_telemetry_counter_rebaseline_pending = True
        self._compaction_telemetry_turn_reset_pending = False

    def _reset_compaction_progress(self) -> None:
        """Reset process-local compaction markers for a fresh/unproven session."""
        self._last_compacted_store_id = 0
        self._ingest_cursor = 0
        self._ingest_cursor_needs_reconcile = False
        self._last_ingest_reconciliation = {"action": "none", "reason": "not run"}
        self._last_compaction_source = None

    def _reset_session_scoped_runtime_state(self) -> None:
        """Reset all session-scoped runtime state.

        Calls both _reset_session_counters and _reset_compaction_progress.
        Proven carry-over paths must restore/advance state from the verified
        source lifecycle after rebinding.
        """
        self._reset_session_counters()
        self._reset_compaction_progress()
        self._generated_ignored_active_replay_placeholder_hashes = set()
        self._generated_ignored_active_replay_placeholder_message_ids = set()
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget = {}
        self._compression_boundary_active_placeholder_digest_ordinals = {}
        self._compression_boundary_stored_placeholder_digest_counts = {}

    def _compaction_source_digest(self, message) -> str:
        identity = self._message_replay_identity(message)
        return hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()

    def _record_compaction_source(self, messages) -> None:
        """Remember the transcript a successful compress() just consumed.

        Hermes may deliver that same pre-compaction transcript to session-end
        observers while it commits the compaction. Every row in it is already
        durable, and the cursor now points into the compacted context, so it
        must be recognised rather than ingested again. Only per-message
        digests are kept, not message content.
        """
        self._last_compaction_source = (
            str(self._session_id or ""),
            [self._compaction_source_digest(message) for message in messages],
        )

    def _compaction_source_suffix(self, session_id, messages):
        """Return the new tail of ``messages`` past the last compaction source.

        ``None`` means ``messages`` is not a replay of the recorded source (the
        normal ingest path applies). An empty list means it is exactly that
        source. The prefix must match element by element; content that only
        resembles earlier history is never skipped.
        """
        recorded = getattr(self, "_last_compaction_source", None)
        if not recorded or recorded[0] != session_id or not messages:
            return None
        source_digests = recorded[1]
        if not source_digests or len(messages) < len(source_digests):
            return None
        for digest, message in zip(source_digests, messages):
            if self._compaction_source_digest(message) != digest:
                return None
        return list(messages[len(source_digests):])

    def _ingest_compaction_source_suffix(self, messages) -> None:
        """Store only the messages appended after the recorded compaction source."""
        suffix_start = len(self._last_compaction_source[1])
        saved_cursor = self._ingest_cursor
        saved_reconcile = self._ingest_cursor_needs_reconcile
        self._ingest_cursor = suffix_start
        self._ingest_cursor_needs_reconcile = False
        try:
            self._ingest_messages(messages)
        finally:
            # The host continues from the compacted context, so keep the cursor
            # positioned there; the next ingest resumes from the compacted list.
            self._ingest_cursor = saved_cursor
            self._ingest_cursor_needs_reconcile = saved_reconcile

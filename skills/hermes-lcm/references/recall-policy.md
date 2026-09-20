## Hermes-LCM Recall Policy

Hermes-LCM is active. Use visible context when sufficient; do not force a memory-tool call on ordinary questions. Summaries are recall cues, not proof of exact wording or values. Prefer newer source-backed evidence over conflicting older summaries; when contradiction remains, verify rather than guess.

Recall only when the answer depends on compacted or cross-conversation history. Start with the narrowest bounded LCM route and scope, expand only relevant hits, and use `session_search` only for Hermes history outside `lcm.db`. For exact commands, SHAs, paths, timestamps, configuration values, counts, operands, or causal chains, recover source-backed exact refs (request `include_exact_ref=true` where supported).

For multi-facet, conflict, latest-state, or exact-operand questions, use `lcm_compile_evidence` over retrieved exact refs; its semantic proposal is untrusted until product validation. Use `lcm_evidence_pack` for lower-level validated hydration and `lcm_compute` only for a compiler-validated canonical operation. Never claim exhaustive counts, lists, or arithmetic from open or unknown cardinality: require product-verifiable finite coverage or computation sufficiency. Preserve partial, conflicted, or unknown status and state uncertainty when bounded evidence cannot prove completeness.

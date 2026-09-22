# Optional supervision owner contract

No provider is imported or discovered. Registration captures the public runtime
`PluginContext.supervision` facade. Disabled, unavailable, unnegotiated, invalid,
expired or stale decisions preserve the existing output; no late hints are saved.

## Host API

`negotiate("supervision.v1") -> Mapping` must return `version="supervision.v1"`,
`supported=True`, and `owner_capabilities` containing only currently active,
authorized owner methods. Merely installing the host facade is not availability.

The synchronous methods are:

- `rank_candidates(request: Mapping) -> Mapping | None`
- `select_windows(request: Mapping) -> Mapping | None`
- `evaluate_relation(request: Mapping) -> Mapping | None`

Requests contain `protocol`, `owner`, `event`, invocation-local `request_id`,
absolute monotonic `deadline`, `facts`, and `completeness`. The host resolves the
current authentic revision, applies profile/source-data policy, and clamps to the
**existing shared tool-round deadline**. An owner can shorten but never renew it.
There is no request-text consent or plugin-registry RPC. Provider work must not
hold an execution/registry lock. Bounded local worker capacity remains occupied
until a timed-out worker actually returns; late responses cannot publish a view.

Responses must echo `request_id`. Rank responses contain `candidate_ids` as an
exact permutation of the supplied head and optionally `conflict_ids` drawn from
that head. Window responses contain `window_ids`. History responses contain one
`candidate_id` and `relation="states_missing_decision"`; the judgment provider
must evaluate the selected hit's own evidence, not another hit's probability.
The host rejects revision changes before returning any applied decision. These
are preference descriptors, not replacement evidence, success outcomes or proof
of complete coverage.

Each projection contains at most eight candidates/windows with <=1200-character
excerpts. Oversized individual excerpts abstain rather than clipping qualifiers.

## LCM ownership

`tools._lcm_recall_rerank(..., supervision=None, scope=None, completeness=None)`
selects one rank owner. An actively negotiated semantic owner replaces Voyage;
abstention/timeout does not invoke Voyage afterward. When capability is absent,
the original Voyage/default-disabled code path is unchanged. Reordering retains
the original source objects, fusion/final scores, exact refs, roles, lineage and
the entire tail. Existing diversity, reference-strict, finite-coverage and
computation validators still run downstream.

The pre-LLM owner consumes only explicit `missing_history_slot` and
`supervision_deadline` hook state. The slot has `slot_id`, `question`, up to eight
`hits=[{exact_ref, excerpt}]`, `visible_refs`, and `expansion_budget=1`. It is not a
new model tool argument. The owner validates exact bytes against its own store
and existing current-conversation lineage. Cross-conversation candidates,
fabricated excerpts and already visible refs abstain. One selected ref goes
through existing `lcm_expand` (600-token bound), with live row/scope checks before
expansion and byte/role checks afterward. No search or scope widening occurs.
Each slot is charged once per runtime session, with a bounded 32-slot fail-closed
ledger; consumed slots do not generate repeated expansions. No completeness or
computation claim is inferred from the expansion. Explicit refs should continue
to call `lcm_expand` directly, without a semantic choice.

LCM clones retain the facade, not another agent's recovery budget. The facade
must resolve runtime context per invocation; it must not capture registration-
time agent state. This owner contract requires host mapping conversion and
active owner-capability negotiation; typed host records alone are not a mapping
adapter. End-to-end host/provider activation is a separate integration gate.

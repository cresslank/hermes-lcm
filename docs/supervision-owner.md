# Optional supervision owner contract

The separate, default-off [literal-record source contract](literal-record.md)
qualifies original structured assertions at ordinary pack/compile hydration.
It does not reinterpret the ranking/recovery facts below as truth or coordinate
authority, and makes no inference calls.

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
- `expand_one_owned_ref(request: Mapping) -> Mapping | None`
- `acknowledge_owner(acknowledgment: Mapping) -> Mapping | None`

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

The ordinary producer is the existing `lcm_grep` tool's optional
`missing_decision` argument: an explicit caller request, never a source/prose
heuristic. It runs the original search once, in its unchanged current/history
scope, then offers at most eight of that result's existing message IDs. Native
request assembly supplies exact visibility; missing visibility, already visible
source text/refs and foreign sessions abstain. There is no extra generated query,
search or broad-scope fallback. The pre-LLM compatibility owner can also consume
explicit `missing_history_slot` and `supervision_deadline` hook state. The slot has `slot_id`, `question`, up to eight
`hits=[{exact_ref, excerpt}]`, `visible_refs`, and `expansion_budget=1`. It is an internal record, not a
caller-supplied source/grant map. The owner validates exact bytes against its own store
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

## Native domain projection

The native facade advertises `owner_deadline=True`: call it on the authenticated
execution thread, not a private worker. It owns the single original <=150 ms
round token and schedules provider I/O elsewhere. Native local observation needs
`history_excerpt`; remote disclosure separately requires explicitly configured
per-field/source policy. Neither the adapter nor source text grants disclosure.

F14 emits canonical `id`, `ref`, `version`, `provenance`, `baseline_ids`,
`needs_triage`, `exact_answer_complete` and candidate facts. A batch fitting the
8-candidate evidence head is not itself a triage opportunity. For an existing
message candidate, LCM may hydrate its whole native row only when <=1200 chars,
with matching session and role; it rechecks the row before returning the view.
This proves no source qualifiers were clipped, not truth/freshness. Retrieval
scope (`all` or matching `current`) establishes the retrieval constraint only;
it never certifies semantic entity/date agreement. Other snippets remain unknown.
The owner does not have an already-certified complete answer at this retrieval
stage; downstream exact/finite-coverage/computation validators remain unchanged.

F15's ordinary producer uses `temporal_contract=historical_source_v1`: a hash of
the current exact source row and `temporal_status=unknown`. That pin establishes
source identity/currency, not that a historical decision remains current or was
never superseded. Older host-owned slots may still carry independently supplied
`current=True/superseded=False`; absence never manufactures those assertions.
The selected hit must pass its own fit judgment. Expansion uses the dedicated
literal `supervision.exact-expansion.v1` codec/action/grant, never a relation alias.
Before and after the existing exact reader, LCM compares source bytes, session,
role, lineage, source, offsets and exact ref under the original deadline. One
slot is charged at most once even when postvalidation rejects the read.

Selection remains `accepted/owner_selected` until the adapter validates and
acknowledges its actual JSON result using `supervision.owner-consumption.v1`.
Changed rows, oversized views or failed exact reads send an explicit rejection;
an applied receipt must never precede that veto. Receipt storage belongs to the
host's shared settlement seam, not LCM. No acknowledgment claims latest-state
truth, completeness, or successful external inference.

For rank selections, `decision_adapter.SelectedRank` is invocation-local pending
state, not an applied permutation. `recall_response.shape_recall_response` runs
the unchanged diversity, hydration, exact-ref/delta and serialization validators
for both baseline and candidate views. Only a surviving selected winner and a
changed delivered identity order/subset can consume the preference. All-seen,
empty, dropped-winner, cap/serialization failures and final authority vetoes
return the ordinarily shaped baseline. The rank digest is SHA-256 of the exact
UTF-8 serialized recall response bytes; nothing is reshaped after that ack.
The baseline comparison adds bounded local shaping work, never another search,
judgment or deadline. Disabled/non-native output shaping remains unchanged.

Native history recovery requires `begin_history_expansion(selection)` from the
host (`history_read_fence=supervision.history-read.v1`). This one-use owner-only
lease spans only the existing bounded local reader, serializing grant revocation
and instruction revisions at the read boundary, not merely at post-read ack.
Missing native support rejects the recovery rather than performing an unfenced
read. The original post-read validators and deadline still apply. Structured
model-visible tool arguments count toward exact-ref availability; hidden reasoning
does not. Known exact refs remain on direct `lcm_expand` without judgment.

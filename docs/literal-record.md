# `lcm.literal-record.v1`: original source assertions, not truth

This new opt-in source grammar is deliberately narrower than ordinary LCM
retrieval or V4 assertion extraction. A record must be **one whole original
stored user/tool message** (including all JSON whitespace), at most 2400 UTF-8
bytes, with exactly these fields:

```json
{
  "schema": "lcm.literal-record.v1",
  "entity": {"namespace": "inventory", "id": "A"},
  "predicate": {"namespace": "metric", "id": "mass"},
  "time": {
    "basis": "event_interval",
    "start": "2026-01-01T00:00:00Z",
    "end": "2026-02-01T00:00:00Z"
  },
  "scope": {
    "namespace": "lab", "id": "L",
    "conditions": [{"namespace": "environment", "id": "dry", "value": true}]
  },
  "quantity": {"kind": "quantity", "unit": "kg"},
  "value": 3,
  "polarity": "positive",
  "modality": "asserted"
}
```

## Closed rules

- Duplicate/unknown/missing keys at any level reject. Prose wrappers, multiple
  records, partial spans, null-as-unknown coordinates, nonfinite numbers and
  oversized records reject. No truncated record can qualify.
- Exact namespace and ID strings match `[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}`.
  Case, namespace and spelling are significant. No aliases/pronouns/renames.
- `event_interval` has nonempty ordered bounds, each an ISO timestamp at second
  precision with explicit `Z` or `±HH:MM` offset. Unknown `-00:00`, missing zones,
  dates alone, fractional precision, invalid dates and reversed/equal bounds
  reject. The v1 interval is half-open `[start,end)`. Minimum pair eligibility
  requires identical literal bounds; no normalization, overlap or union.
- `atemporal` explicitly requires `start:null,end:null`. It is not a fallback
  for missing event time. Observation/ingestion timestamps remain attribution
  only and cannot manufacture temporal coordinates.
- Scope has an explicit ordered list of 0–8 equality conditions. Each condition
  has exactly `namespace,id,value`; duplicate condition identities reject.
  Nested/free-prose qualifier maps are unsupported. The entire ordered scope
  must match for a later minimum comparison; empty conditions explicitly means
  no listed constraints, not inferred global completeness.
- `quantity.kind` is `quantity` or `non_quantity`. Quantities require a numeric
  value and an exact unit from `1,count,m,s,kg,K,USD,EUR,percent`.
  Non-quantities explicitly require `unit:null`. Empty/missing/unknown units
  reject. No conversions or equivalent-spelling guesses occur.
- Values/condition values support booleans, nonempty strings of at most 256
  characters without control characters, or finite numbers of absolute value
  at most `2**53 - 1`. The original numeric token must round-trip without
  changing its numerical meaning through the existing canonical JSON encoder.
  Bounds are checked before conversion; underflow and lossy decimal tokens
  abstain (for example `1e-400`, `0.100000000000000000001`, and
  `9007199254740991.1`). Ordinary `0.1`, `1.25`, and in-bound integers remain
  supported. This applies equally to record values and scope conditions; it
  introduces no arbitrary-precision coordinate type. Null/list/map values are
  unknown/unsupported and abstain.
- Polarity is `positive|negative`; modality is `asserted|possible|necessary`.
  Their presence describes the original assertion, not model confidence.

The constants above designate a syntax, not an ontology of true predicates.
The source can lie. The validator establishes only what this original source
asserted under these explicit qualifiers. Source content supplies no action,
permission, delivery destination, or host authority.

## Native publication

The ordinary plugin entrypoint installs `literal_source.install_literal_source_owner`
only after context-engine registration. It requires a host supporting the
explicit `supervision.literal-sources.v1` protocol and opted-in host profile
source-recipient grants. Old/disabled hosts retain ordinary tool behavior.

The actual `lcm_evidence_pack` and `lcm_compile_evidence` calls capture valid
rows at `_resolve_candidate`; auto compilation separately captures at
`requirements_compiler._normalize_ref`. A `ContextVar` prevents another tool or
engine from adding records to the invocation. Selection entity/value/unit/date,
model predicates/scopes, assertion IDs, and assistant transformations never
supply the literal coordinates. No assertions are changed or promoted.

The adapter retains immutable `LiteralRecord` records, then asks the native host
to publish their opaque owner handles. The host re-resolves through the actual
registered provider, not a copied result. Each revalidation performs a point read
of the same selected SQLite row, revalidates its full original grammar, and
compares original bytes, span, role, session, source and attribution/version.
There is no additional retrieval, scan, inference, credential resolution or
provider discovery. This path makes **zero inference calls**.

The host stamps owner generation, exact native revision/profile, invocation and
original shared deadline. At most 8 records qualify per publication and 64 are
retained per registration; overflow abstains instead of claiming a complete
prefix. Output may add `source_propositions` opaque refs without changing any
public tool argument schema. A copied ref/payload is not owner authority.
The provider pins active engine/store/home/database and session binding. Store
rebinding, changed rows, revoked registration/grants and expired/stale invocations
fail closed. Lookup is recipient-specific; another plugin cannot borrow a grant.
The native invocation also pins a per-engine lifecycle generation. Session
start/end/reset/shutdown invalidate it before mutation; only a successful start
reopens capture. A→B→A cannot revive an old invocation. Publication and lookup
accept under the same short provider fence used by lifecycle invalidation, after
row reads finish outside host and provider locks. A fresh legitimate invocation
can publish after rebinding, but cannot renew the original shared deadline.
Older host adapters without the coordinated fence retain ordinary tool output
without native source publication.

The native host's `SourcePropositionV1` contains `LiteralSourceRecordV1` with
`schema_id,exact_ref,row_hash,record_span,source_bytes,coordinate_pins,source_attribution`.
The original bytes are exact UTF-8 of stored content; spans use Unicode character
offsets, as existing LCM exact refs do. Coordinates/attribution use immutable
pairs with canonical JSON values. Native metadata nulls remain null. The row
hash pins the fields listed by `literal_record.ROW_FIELDS`; native attribution
is separately bounded to 4096 bytes. Hashes provide integrity, not truth.

These are bounded in-memory, invocation/revision/deadline-bound prerequisites,
not durable accepted claims. F22 still needs the host adoption/dependency graph,
comparison/action codecs, correction policy and real sink emission joins. It is
not complete merely because original source records can now be published.

## Tests

`tests/test_literal_record.py` uses real isolated SQLite rows for grammar,
provenance, immutable-coordinate and absent-capability controls. Existing
`test_evidence_pack.py`, `test_evidence_compiler.py` and `test_evidence_contract.py`
qualify the unchanged tools. Host `test_supervision_literal_sources.py` provides
the producer proof: actual plugin entrypoint/clone, actual ordinary pack,
proposal and auto tool dispatch, native facade registration/publication/lookup,
real SQLite, source/recipient mutations and zero-network/inference guards.
No handwritten source proposition stands in for that vertical.

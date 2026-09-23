# Dependency assurance contract

Hermes-LCM uses a **host-owned dependency boundary**. It is installed as source
inside Hermes Agent rather than as an independently resolved Python package, so
Hermes Agent owns the environment, lock, upgrades, and any vulnerability scan of
the resolved environment. Hermes-LCM must not silently create a second resolver
or install packages to make an assurance tool pass.

The authoritative, versioned contract is
[`dependency-contract.json`](../dependency-contract.json). Contract version
`1.0.6` supports:

- Hermes Agent `>=0.16,<1`
- Python 3.11, 3.12, 3.13, and 3.14 (the CI matrix)
- the required `agent` host API
- explicitly listed compatibility-path and optional feature imports, with the
  imported API each host-resolved version must provide

The Hermes-LCM maintainers own contract updates. Increment the contract version
and review the boundary whenever a scanned runtime import, supported Python or
Hermes Agent version, or required imported API changes.

The redaction stress benchmark additionally needs `cryptography` for disposable
Ed25519 fixture generation. It is imported only when that scenario runs, never
by ordinary plugin loading. CI and the documented test setup install it; missing
support fails the scenario rather than skipping redaction coverage.

## Optional native supervision APIs

The `agent` module remains required for the baseline plugin. Its API inventory
also includes feature-scoped host integrations already used by the runtime:

- `literal_source.py` imports `LiteralSourceProviderV1` and
  `LiteralSourceRecordV1` only when a supervision facade is supplied. Missing
  imports, a missing `literal_source_binding_fence`, or a missing facade
  registration method leave native publication unregistered.
- `literal_source_final.py` consumes `NativeFinalUse`, `_FinalPhase`, and
  `selection_scope` inside registered source-owner callbacks.
- `literal_source_correction.py` consumes `_Attempt`, `Review`, `enabled`,
  `source_for`, `candidate_received`, `get_active_subagent_parent`, and
  `comparable`; `literal_source_events.py` also consumes `comparable` and
  `source_event_received`. These imports support native correction custody,
  candidate hydration, and postcommit source-event delivery.
- `decision_adapter.py` negotiates retrieval presentation using `VERSION` and
  `negotiated`, then consumes `validate` and `annotate` only for that output
  contract. Missing negotiation imports retain order-only behavior; presentation
  or settlement failures return the ordinary shaped baseline.

Final-use and correction imports are not individually guarded by `ImportError`
in their callbacks. A host enabling those features must supply the matching API
set, including the private `_FinalPhase` and `_Attempt` types; the broad baseline
host version range alone does not establish feature compatibility. Correction
source-event preparation and postcommit delivery have separate exception guards
so optional metadata or callback failures do not strand native source writes.
The inventory was checked against host commit
`bd6e7e754c6372007a93acf6573dc85d27253e09`; this is source-level compatibility
evidence, not a claim that every supported host version supplies these APIs.

## Mechanical validation

Run:

```bash
python scripts/validate_dependency_contract.py --report-environment
```

The validator uses only the Python standard library. It:

1. validates the contract schema, ownership, version policies, and non-empty API
   requirements;
2. checks that the supported Python versions exactly match the CI matrix;
3. parses shipped plugin, operator-script, and benchmarking Python sources with
   `ast`, including literal `importlib.import_module(...)` and `__import__(...)`
   calls;
4. fails on undeclared external imports and collector-observed imported APIs that lack an `imported_api` declaration; it does not reject an `imported_api` declaration solely because the current collector did not observe it; and
5. reports locally available modules and distribution versions literally.

The machine-readable `imported_api_validation` policy is `observed-coverage-only`. It guarantees coverage of the external imports and imported APIs the current static collector observes. Declarations may conservatively over-approximate that observed set, including APIs reached through parameter propagation, returned capabilities, runtime wiring, or other patterns the collector does not fully model.

This validator is not an SBOM, lockfile, dependency resolver, complete runtime-reachability analysis, or proof that every declared API is currently reachable. Removing a use does not automatically establish that a declaration is stale; contract changes remain a maintainer-reviewed, versioned decision.

The same command is a CI gate and a release-validation gate. The environment
report is **not a vulnerability scan**. It does not prove that an installed
version is vulnerability-free, and source checkouts may have no discoverable
distribution version.

## Resolver and scanner responsibility

For a release candidate, run the host repository's authoritative lock or SBOM
validation and whichever dependency scanner is actually available against that
resolved Hermes Agent environment. Record the exact host commit/lock identity,
scanner name/version, command, and output. If no scanner is available, record
that absence; do not report a clean CVE result.

Optional imports remain feature-scoped and fail closed or fall back as documented
in the contract. Adding an optional package solely to satisfy this validator is
not an acceptable fix.

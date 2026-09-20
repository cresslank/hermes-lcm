from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_COMPACT_RECALL_POLICY = """## Hermes-LCM Recall Policy

Hermes-LCM is active. Use visible context when sufficient; do not force a memory-tool call on ordinary questions. Summaries are recall cues, not proof of exact wording or values. Prefer newer source-backed evidence over conflicting older summaries; when contradiction remains, verify rather than guess.

Recall only when the answer depends on compacted or cross-conversation history. Start with the narrowest bounded LCM route and scope, expand only relevant hits, and use `session_search` only for Hermes history outside `lcm.db`. For exact commands, SHAs, paths, timestamps, configuration values, counts, operands, or causal chains, recover source-backed exact refs (request `include_exact_ref=true` where supported).

For multi-facet, conflict, latest-state, or exact-operand questions, use `lcm_compile_evidence` over retrieved exact refs; its semantic proposal is untrusted until product validation. Use `lcm_evidence_pack` for lower-level validated hydration and `lcm_compute` only for a compiler-validated canonical operation. Never claim exhaustive counts, lists, or arithmetic from open or unknown cardinality: require product-verifiable finite coverage or computation sufficiency. Preserve partial, conflicted, or unknown status and state uncertainty when bounded evidence cannot prove completeness."""
EXPECTED_COMPACT_RECALL_POLICY_SHA256 = (
    "f8a165efe8f19a9f8733ffdfd0a96225dc53e84d012db02815b2955dbd46bbe8"
)


def _load_guidance_module():
    package_name = "hermes_lcm_recall_guidance"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        str(REPO_ROOT / "__init__.py"),
        submodule_search_locations=[str(REPO_ROOT)],
    )
    package = importlib.util.module_from_spec(package_spec)
    assert package_spec is not None
    assert package_spec.loader is not None
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)
    return __import__(f"{package_name}.guidance", fromlist=["guidance"])


def _load_schemas_module():
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm_recall_schemas", REPO_ROOT / "schemas.py"
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_recall_policy_is_canonical_compact_and_benchmark_neutral():
    guidance = _load_guidance_module()
    first = guidance.get_recall_policy()
    second = guidance.get_recall_policy()

    assert first == second
    assert first == guidance.RECALL_POLICY_PATH.read_text(encoding="utf-8").strip()
    assert first == EXPECTED_COMPACT_RECALL_POLICY
    assert len(first.encode("utf-8")) == 1338
    assert guidance.MAX_RECALL_POLICY_BYTES == 8 * 1024
    assert guidance.recall_policy_sha256() == EXPECTED_COMPACT_RECALL_POLICY_SHA256
    lowered = first.lower()
    for forbidden in ("longmemeval", "question_id", "reference answer", "judge output"):
        assert forbidden not in lowered


def test_recall_policy_preserves_exact_evidence_and_completion_gates():
    guidance = _load_guidance_module()
    policy = guidance.get_recall_policy()

    for phrase in (
        "do not force a memory-tool call on ordinary questions",
        "Summaries are recall cues, not proof of exact wording or values",
        "Prefer newer source-backed evidence",
        "narrowest bounded LCM route and scope",
        "`session_search` only for Hermes history outside `lcm.db`",
        "recover source-backed exact refs",
        "`include_exact_ref=true`",
        "`lcm_compile_evidence` over retrieved exact refs",
        "semantic proposal is untrusted until product validation",
        "`lcm_evidence_pack` for lower-level validated hydration",
        "`lcm_compute` only for a compiler-validated canonical operation",
        "open or unknown cardinality",
        "product-verifiable finite coverage or computation sufficiency",
        "Preserve partial, conflicted, or unknown status",
        "state uncertainty when bounded evidence cannot prove completeness",
    ):
        assert phrase in policy


def test_detailed_routing_moved_only_to_discoverable_reference_and_schemas():
    guidance = _load_guidance_module()
    policy = guidance.get_recall_policy()
    assert "1-3 distinctive terms" not in policy
    assert "`lcm_describe`" not in policy
    assert "`lcm_expand_query`" not in policy
    assert "FTS5" not in policy

    skill_root = REPO_ROOT / "skills" / "hermes-lcm"
    recall_reference = (skill_root / "references" / "recall-tools.md").read_text(
        encoding="utf-8"
    )
    for phrase in (
        "Prefer 1-3 distinctive terms",
        "Recommended current-session escalation",
        "Follow each result's `expand_hint`",
        "Use Hermes `session_search`",
        "Continue with `after_store_id`",
        "Set `include_exact_ref=true`",
    ):
        assert phrase in recall_reference

    schemas = _load_schemas_module()

    def _descriptions(value):
        if isinstance(value, dict):
            if isinstance(value.get("description"), str):
                yield value["description"]
            for child in value.values():
                yield from _descriptions(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from _descriptions(child)

    schema_descriptions = "\n".join(
        description
        for schema in (
            schemas.LCM_GREP,
            schemas.LCM_RECALL,
            schemas.LCM_LOAD_SESSION,
            schemas.LCM_EXPAND,
            schemas.LCM_EXPAND_QUERY,
            schemas.LCM_COMPILE_EVIDENCE,
            schemas.LCM_EVIDENCE_PACK,
        )
        for description in _descriptions(schema)
    )
    for phrase in (
        "FTS5 defaults to AND matching, so prefer 1-3 distinctive terms",
        "Search the agent's entire memory across ALL conversations",
        "This is enumeration, not search",
        "Mode selection (exactly one)",
        "Prefer this for questions about the active conversation after compaction",
        "Finite coverage cannot be asserted by the proposal",
        "open cardinality never closes from a caller assertion alone",
    ):
        assert phrase in schema_descriptions


def test_bundled_skill_has_valid_minimal_frontmatter_and_matching_tool_references():
    skill_root = REPO_ROOT / "skills" / "hermes-lcm"
    skill_text = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    assert skill_text.startswith("---\nname: hermes-lcm\ndescription:")
    frontmatter = skill_text.split("---", 2)[1]
    keys = {
        line.split(":", 1)[0].strip()
        for line in frontmatter.splitlines()
        if ":" in line
    }
    assert keys == {"name", "description"}

    recall_reference = (skill_root / "references" / "recall-tools.md").read_text(
        encoding="utf-8"
    )
    documented = {
        "lcm_grep",
        "lcm_recall",
        "lcm_recent",
        "lcm_load_session",
        "lcm_describe",
        "lcm_expand",
        "lcm_expand_query",
        "lcm_compute",
        "lcm_compile_evidence",
        "lcm_evidence_pack",
    }
    for tool_name in documented:
        assert f"`{tool_name}`" in recall_reference

    schemas_text = (REPO_ROOT / "schemas.py").read_text(encoding="utf-8")
    for tool_name in documented:
        assert f'"name": "{tool_name}"' in schemas_text

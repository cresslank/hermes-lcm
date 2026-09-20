"""One tool surface for host registration, engine schemas, and dispatch.

Eligibility follows the initialized runtime resource, not a second copy of the
configuration flags. Disabled tools remain directly callable for diagnostics,
but are neither advertised nor allowed to ingest a supplied live transcript.
"""
from dataclasses import dataclass
from typing import Any

from . import schemas


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    schema: dict[str, Any]
    emoji: str
    runtime_attribute: str = ""
    enable_with: str = ""

    def eligible(self, engine: Any) -> bool:
        return not self.runtime_attribute or getattr(engine, self.runtime_attribute, None) is not None

    def disabled_response(self, engine: Any) -> dict[str, Any] | None:
        if self.eligible(engine):
            return None
        return {
            "status": "disabled",
            "reason": f"{self.name} is disabled for this runtime",
            "enable_with": self.enable_with,
        }

    @property
    def handler(self):
        # Resolve lazily: bootstrap imports modules in varying orders, and tests
        # may instrument a handler after the engine was constructed.
        from . import tools
        return getattr(tools, self.name)


TOOL_DESCRIPTORS = (
    ToolDescriptor("lcm_grep", schemas.LCM_GREP, "🔍"),
    ToolDescriptor("lcm_recall", schemas.LCM_RECALL, "🧠"),
    ToolDescriptor("lcm_query_state", schemas.LCM_QUERY_STATE, "🧾", "_assertions", "LCM_ASSERTIONS_ENABLED=true"),
    ToolDescriptor("lcm_compute", schemas.LCM_COMPUTE, "🧮"),
    ToolDescriptor("lcm_compile_evidence", schemas.LCM_COMPILE_EVIDENCE, "🧷"),
    ToolDescriptor("lcm_evidence_pack", schemas.LCM_EVIDENCE_PACK, "📦"),
    ToolDescriptor("lcm_retrieve", schemas.LCM_RETRIEVE, "🧭", "_adaptive_retrieval", "LCM_ADAPTIVE_RETRIEVAL_ENABLED=true"),
    ToolDescriptor("lcm_recent", schemas.LCM_RECENT, "🕒"),
    ToolDescriptor("lcm_load_session", schemas.LCM_LOAD_SESSION, "📋"),
    ToolDescriptor("lcm_describe", schemas.LCM_DESCRIBE, "📊"),
    ToolDescriptor("lcm_expand", schemas.LCM_EXPAND, "🔎"),
    ToolDescriptor("lcm_expand_query", schemas.LCM_EXPAND_QUERY, "❓"),
    ToolDescriptor("lcm_status", schemas.LCM_STATUS, "💚"),
    ToolDescriptor("lcm_inspect", schemas.LCM_INSPECT, "🧭"),
    ToolDescriptor("lcm_doctor", schemas.LCM_DOCTOR, "🏥"),
)
TOOLS_BY_NAME = {descriptor.name: descriptor for descriptor in TOOL_DESCRIPTORS}


def eligible_tools(engine: Any):
    return tuple(descriptor for descriptor in TOOL_DESCRIPTORS if descriptor.eligible(engine))

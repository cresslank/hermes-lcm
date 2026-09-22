"""Provider-neutral, presentation-only evidence decisions owned by LCM.

The optional runtime ``PluginContext.supervision`` facade implements synchronous
``rank_candidates(request)``, ``select_windows(request)`` and
``evaluate_relation(request)`` -> Mapping | None. It authenticates the current
revision, source-data policy and shared round deadline; source text NEVER grants
network consent. This module does not import or discover any judgment provider.

Requests use supervision.v1, an invocation-local request_id, absolute monotonic
deadline, owner/event, facts and completeness. Responses echo request_id and
contain only candidate_ids (rank) or candidate_id + relation (history). The
facade must reject stale revisions. Detached projections cannot mutate evidence.
LCM revalidates IDs and exact bytes; no decision changes coverage or computation.
"""
from __future__ import annotations

import copy
import contextvars
import hashlib
import json
import math
import re
import time
import threading
import uuid
from collections.abc import Mapping

_RECOVERY_LOCK = threading.Lock()
_MAX_CANDIDATES = 8
_MAX_EXCERPT = 1200
_EXACT = re.compile(r"lcm:([1-9][0-9]*):([0-9]+)-([0-9]+)\Z")


def supports(facade, method: str) -> bool:
    """Negotiate active owner capability; an installed but inactive host is baseline."""
    if facade is None or not callable(getattr(facade, method, None)):
        return False
    try:
        capability = facade.negotiate("supervision.v1")
        return (isinstance(capability, Mapping) and capability.get("supported") is True
                and capability.get("version") == "supervision.v1"
                and method in capability.get("owner_capabilities", ()))
    except Exception:
        return False


def admission_deadline(operation_deadline: float) -> float:
    """One owner admission deadline; never refresh it between sibling decisions."""
    if not math.isfinite(operation_deadline):
        return time.monotonic()
    return min(operation_deadline, time.monotonic() + 0.150)


def _request(facade, method: str, *, event: str, facts: dict,
             completeness: dict, deadline: float):
    if not supports(facade, method):
        return None
    if time.monotonic() >= deadline:
        return None
    from .tools import _run_within_deadline

    request_id = uuid.uuid4().hex
    request = {"protocol": "supervision.v1", "owner": "lcm", "event": event,
               "request_id": request_id, "deadline": deadline,
               "facts": copy.deepcopy(facts), "completeness": copy.deepcopy(completeness)}
    context = contextvars.copy_context()
    try:
        if facade.negotiate("supervision.v1").get("owner_deadline") is True:
            # Native host owns the bounded wait and authenticates this execution
            # thread. A private worker is not an execution-owner capability.
            response = getattr(facade, method)(request)
        else:
            response = _run_within_deadline(
                lambda: context.run(getattr(facade, method), request),
                remaining_s=deadline - time.monotonic(), name="lcm-owner-decision",
            )
        if (time.monotonic() >= deadline or not isinstance(response, Mapping)
                or response.get("request_id") != request_id):
            return None
        return response
    except Exception:
        # A decision failure is neither a retrieval failure nor permission to
        # start another rank provider. Never retain a late hint for another call.
        return None


def rank_candidates(facade, query: str, ordered: list[dict], *, window: int,
                    deadline: float, scope: dict, completeness: dict, engine=None) -> list[dict]:
    """Reorder only an existing head; retain scores, source objects and all tail."""
    head = ordered[:min(window, _MAX_CANDIDATES)]
    if len(head) < 2 or len(ordered) <= len(head) or len(query) > 4096:
        return ordered
    candidates = []
    by_id = {}
    source_rows = {}
    for entry in head:
        hit = entry["hit"]
        excerpt = hit.get("snippet") or ""
        row = None
        if engine is not None and hit.get("kind") == "message_excerpt" and type(hit.get("store_id")) is int:
            row = engine._store.get(hit["store_id"])
            if row and row.get("session_id") == hit.get("session_id") and row.get("role") == hit.get("role"):
                content = row.get("content")
                if isinstance(content, str) and 0 < len(content) <= _MAX_EXCERPT:
                    excerpt = content
                    source_rows[hit["store_id"]] = copy.deepcopy(row)
        # Do not clip qualifiers to qualify an otherwise ineligible projection.
        if not isinstance(excerpt, str) or len(excerpt) > _MAX_EXCERPT:
            return ordered
        source = {key: copy.deepcopy(hit[key]) for key in (
            "kind", "store_id", "node_id", "session_id", "role", "source",
            "timestamp", "chunk_span", "exact_ref", "lineage", "source_ids",
        ) if key in hit}
        identity = hashlib.sha256(json.dumps(
            {"source": source, "excerpt": excerpt}, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False,
        ).encode()).hexdigest()
        if identity in by_id:
            return ordered
        by_id[identity] = entry
        # Only an exact WHOLE native message proves that this projection did
        # not clip any source qualifier. Snippet length/truncation absence does not.
        whole = hit.get("store_id") in source_rows
        in_scope = bool(whole and (scope.get("session_scope") == "all" or
                        (scope.get("session_scope") == "current" and
                         hit.get("session_id") == scope.get("current_session_id"))))
        ref = f"lcm:{hit['store_id']}:0-{len(excerpt)}" if whole else hit.get("exact_ref")
        candidates.append({"id": identity, "candidate_id": identity, "excerpt": excerpt,
                           "ref": ref, "version": hashlib.sha256(excerpt.encode()).hexdigest(),
                           "provenance": "lcm_message" if whole else "lcm_snippet",
                           "constraints_match": True if in_scope else None,
                           "qualifiers_complete": True if whole else None,
                           "instruction_flag": bool(re.search(r"ignore|instruction|system|assistant", excerpt, re.I)),
                           "source": source, "original_rank": len(candidates) + 1})
    response = _request(facade, "rank_candidates", event="retrieval_candidates",
                        facts={"query": query, "candidates": candidates, "scope": scope,
                               "required_ids": list(by_id), "baseline_ids": list(by_id),
                               "needs_triage": len(ordered) > len(head), "exact_answer_complete": False},
                        completeness=completeness, deadline=deadline)
    if response is None:
        return ordered
    ids = response.get("candidate_ids")
    conflicts = response.get("conflict_ids", [])
    if (not isinstance(ids, (list, tuple)) or not all(isinstance(x, str) for x in ids)
            or len(ids) != len(by_id) or set(ids) != set(by_id)
            or not isinstance(conflicts, (list, tuple))
            or not all(isinstance(x, str) and x in by_id for x in conflicts)):
        return ordered
    if engine is not None and source_rows and (time.monotonic() >= deadline or any(
            engine._store.get(sid) != row for sid, row in source_rows.items())):
        return ordered
    # Conflicts remain source evidence, never synthesized text or deleted refs.
    return [by_id[x] for x in ids] + ordered[len(head):]


def recover_missing_history(engine, slot: Mapping, *, deadline: float) -> dict | None:
    """Expand <=1 already-supplied exact hit in the current conversation scope.

    ``slot`` is host-owned missing-history state, not a tool argument: slot_id, question,
    hits [{exact_ref, excerpt}], visible_refs and expansion_budget=1. It cannot
    grant cross-session access. Explicit refs should use lcm_expand directly.
    This consumer performs no search and never changes LCM evidence validators.
    """
    facade = getattr(engine, "supervision", None)
    if (not supports(facade, "evaluate_relation") or not isinstance(slot, Mapping)
            or slot.get("expansion_budget") != 1 or slot.get("explicit_ref_available") is not False):
        return None
    question, hits = slot.get("question"), slot.get("hits")
    slot_id = slot.get("slot_id")
    if not isinstance(slot_id, str) or not 0 < len(slot_id) <= 128:
        return None
    key = (engine.current_session_id, slot_id)
    with _RECOVERY_LOCK:
        recovered = getattr(engine, "_supervision_recovered_slots", set())
        if key in recovered or len(recovered) >= 32:
            return None
    if (not isinstance(question, str) or not 0 < len(question) <= 4096
            or not isinstance(hits, (list, tuple)) or not 1 <= len(hits) <= 8):
        return None
    from .tools import _recent_conversation_scope_session_ids, lcm_expand

    scope = set(_recent_conversation_scope_session_ids(engine))
    visible = slot.get("visible_refs", ())
    if not isinstance(visible, (list, tuple)):
        return None
    rows = {}
    candidates = []
    for hit in hits:
        if (not isinstance(hit, Mapping) or hit.get("current") is not True
                or hit.get("superseded") is not False):
            return None
        ref = hit.get("exact_ref")
        match = _EXACT.fullmatch(ref) if isinstance(ref, str) else None
        if match is None or ref in visible or ref in rows:
            return None
        store_id, start, end = map(int, match.groups())
        row = engine._store.get(store_id)
        if row is None or row.get("session_id") not in scope:
            return None
        content = row.get("content") or ""
        if not (start == 0 < end == len(content) and end - start <= _MAX_EXCERPT):
            return None
        if hit.get("excerpt") != content[start:end]:
            return None
        rows[ref] = (store_id, start, copy.deepcopy(row))
        candidates.append({"id": ref, "ref": ref, "candidate_id": ref, "exact_ref": ref,
                           "authorized": True, "scope": engine.current_session_id,
                           "session": row["session_id"], "current": hit.get("current"),
                           "superseded": hit.get("superseded"),
                           "excerpt_complete": start == 0 and end == len(content),
                           "excerpt": content[start:end], "session_id": row["session_id"],
                           "role": row.get("role"), "source": row.get("source"),
                           "timestamp": row.get("timestamp")})
    response = _request(facade, "evaluate_relation", event="missing_history_slot",
                        facts={"question": question, "missing_slot": question, "candidates": candidates,
                               "visible_refs": list(visible), "recovery_budget": 1,
                               "already_visible": False,
                               "explicit_ref_available": slot.get("explicit_ref_available"),
                               "scope": engine.current_session_id},
                        completeness={"archive_complete": False}, deadline=deadline)
    if response is None or response.get("relation") != "states_missing_decision":
        return None
    selected = response.get("candidate_id")
    if not isinstance(selected, str) or selected not in rows:
        return None
    store_id, start, before = rows[selected]
    # Revalidate live owner scope and exact source bytes after the judgment.
    if (time.monotonic() >= deadline
            or before.get("session_id") not in set(_recent_conversation_scope_session_ids(engine))
            or engine._store.get(store_id) != before):
        return None
    with _RECOVERY_LOCK:
        recovered = getattr(engine, "_supervision_recovered_slots", set())
        if key in recovered or len(recovered) >= 32:
            return None
        engine._supervision_recovered_slots = recovered | {key}
    result = json.loads(lcm_expand({"store_id": store_id, "content_offset": start,
                                   "max_tokens": 600, "include_exact_ref": True}, engine=engine))
    if (time.monotonic() >= deadline or result.get("session_id") != before["session_id"]
            or result.get("role") != before.get("role") or "error" in result
            or result.get("content") != (before.get("content") or "")[start:start + len(result.get("content", ""))]):
        return None
    # The ordinary exact-history reader owns bytes, role, lineage and offsets.
    return result

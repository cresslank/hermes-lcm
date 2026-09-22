"""Provider-neutral, presentation-only evidence decisions owned by LCM.

The optional runtime ``PluginContext.supervision`` facade implements synchronous
``rank_candidates(request)``, ``select_windows(request)`` and
``expand_one_owned_ref(request)`` -> Mapping | None. It authenticates the current
revision, source-data policy and shared round deadline; source text NEVER grants
network consent. This module does not import or discover any judgment provider.

Requests use supervision.v1, an invocation-local request_id, absolute monotonic
deadline, owner/event, facts and completeness. Responses echo request_id and
contain candidate_ids (rank) or candidate_id + relation (history). Negotiated
rank output may also carry bounded conflict/isolation IDs for the final renderer. The
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
    # Optional native output contract; old hosts remain order-only.
    try:
        from agent.supervision_retrieval_presentation import VERSION, negotiated
        if method == "rank_candidates" and negotiated(facade):
            request["output_contract"] = VERSION
    except ImportError:
        pass
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
            if isinstance(response, Mapping):
                acknowledge(facade, response)
            return None
        if response.get("output_contract") != request.get("output_contract"):
            acknowledge(facade, response)
            return None
        return response
    except Exception:
        # A decision failure is neither a retrieval failure nor permission to
        # start another rank provider. Never retain a late hint for another call.
        return None


def acknowledge(facade, response, result=None, *, serialized=False):
    """A selected ID is not a consumed view; postvalidation may still veto it."""
    try:
        if response.get("consumption") != "supervision.owner-consumption.v1":
            capability = facade.negotiate("supervision.v1")
            return isinstance(capability, Mapping) and capability.get("owner_consumption") is None
        encoded = result if serialized else json.dumps(
            result, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        digest = None if result is None else hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        receipt = facade.acknowledge_owner({"request_id": response["request_id"],
            "receipt_id": response["receipt_id"], "candidate_ids": response["candidate_ids"],
            "effect_digest": digest})
        return isinstance(receipt, Mapping) and receipt.get("status") == "applied"
    except Exception:
        return False


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
    blocks = [response.get(k, []) for k in ("conflict_ids", "isolated_ids")]
    if (not isinstance(ids, (list, tuple)) or not all(isinstance(x, str) for x in ids)
            or len(ids) != len(by_id) or set(ids) != set(by_id)
            or any(not isinstance(values, (list, tuple))
                   or any(type(x) is not str or x not in by_id for x in values)
                   or len(set(values)) != len(values) for values in blocks)):
        acknowledge(facade, response)
        return ordered
    if engine is not None and source_rows and (time.monotonic() >= deadline or any(
            engine._store.get(sid) != row for sid, row in source_rows.items())):
        acknowledge(facade, response)
        return ordered
    if (list(ids) == list(by_id) and not (
            response.get("output_contract") == "supervision.retrieval-presentation.v1" and any(blocks))):
        acknowledge(facade, response)
        return ordered
    # Conflicts remain source evidence, never synthesized text or deleted refs.
    result = [by_id[x] for x in ids] + ordered[len(head):]
    if response.get("consumption") == "supervision.owner-consumption.v1":
        return SelectedRank(result, baseline=ordered, facade=facade, response=response,
                            deadline=deadline, source_rows=source_rows, candidates=candidates, by_id=by_id)
    return result if acknowledge(facade, response, result) else ordered


class SelectedRank(list):
    """Invocation-local pending selection, never an applied internal permutation."""
    def __init__(self, entries, *, baseline, facade, response, deadline, source_rows, candidates=(), by_id=None):
        super().__init__(entries)
        self.baseline = baseline
        self.facade, self.response = facade, response
        self.deadline, self.source_rows = deadline, source_rows
        self.candidates, self.by_id = candidates, by_id or {}


def finish_rank(selection, shape, args):
    """Settle only the exact UTF-8 tool response after all ordinary shaping.

    Both views use the same validators. A dropped winner, unchanged delivery,
    serialization failure or revoked/expired grant returns the shaped baseline.
    No retrieval, new decision, or renewed deadline is introduced here.
    """
    from .tools import _hit_identity, _LCM_RECALL_RESPONSE_CHAR_CAP
    try:
        baseline = shape(ordered=copy.deepcopy(selection.baseline), rerank_status="disabled",
                         **{**args, "summary_leads": copy.deepcopy(args["summary_leads"])})
        encoded = shape(ordered=copy.deepcopy(list(selection)), rerank_status="applied",
                        **{**args, "summary_leads": copy.deepcopy(args["summary_leads"])})
        hits = json.loads(encoded)["hits"]
        original_hits = json.loads(baseline)["hits"]
        winner = _hit_identity(selection[0]["hit"])
        effective = (hits and winner == _hit_identity(hits[0]) and
                     [_hit_identity(h) for h in hits] != [_hit_identity(h) for h in original_hits])
        if selection.response.get("output_contract") == "supervision.retrieval-presentation.v1":
            from agent.supervision_retrieval_presentation import annotate, negotiated, validate
            if not negotiated(selection.facade):
                raise ValueError("presentation_downgrade")
            conflicts, isolated = validate(selection.response, selection.by_id)
            if conflicts or isolated:
                # Annotation cannot make a source lost by diversity/hydration
                # appear preserved. Keep every baseline row and flagged qualifier.
                final_by_key = {_hit_identity(h): (i, h) for i, h in enumerate(hits)}
                if any(final_by_key.get(_hit_identity(h), (None, None))[1] != h for h in original_hits):
                    raise ValueError("presentation_source_loss")
                locations = {}
                for candidate in selection.candidates:
                    key = _hit_identity(selection.by_id[candidate["id"]]["hit"])
                    index, hit = final_by_key.get(key, (None, None))
                    locations[candidate["id"]] = (f"#/hits/{index}", candidate["ref"])
                    if candidate["id"] in (*conflicts, *isolated) and (
                            hit is None or hit.get("content", hit.get("snippet")) != candidate["excerpt"]):
                        raise ValueError("presentation_qualifier_loss")
                document = annotate(json.loads(encoded), selection.response, locations)
                encoded = json.dumps(document, ensure_ascii=False, allow_nan=False)
                effective = True
        current = (time.monotonic() < selection.deadline and all(
            args["engine"]._store.get(sid) == row for sid, row in selection.source_rows.items()))
        within_cap = len(encoded) <= _LCM_RECALL_RESPONSE_CHAR_CAP
        if effective and current and within_cap:
            if acknowledge(selection.facade, selection.response, encoded, serialized=True):
                return encoded
        else:
            acknowledge(selection.facade, selection.response)
        return baseline
    except Exception:
        acknowledge(selection.facade, selection.response)
        # Preserve ordinary baseline errors rather than manufacture an empty success.
        return shape(ordered=copy.deepcopy(selection.baseline), rerank_status="disabled", **args)


def recover_missing_history(engine, slot: Mapping, *, deadline: float) -> dict | None:
    """Expand <=1 already-supplied exact hit in the current conversation scope.

    ``slot`` is owner-created missing-history state, not a source/grant tool argument:
    slot_id, question, hits [{exact_ref, excerpt}], visible_refs and expansion_budget=1. It cannot
    grant cross-session access. Explicit refs should use lcm_expand directly.
    This consumer performs no search and never changes LCM evidence validators.
    """
    facade = getattr(engine, "supervision", None)
    if (not supports(facade, "expand_one_owned_ref") or not isinstance(slot, Mapping)
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
        if (not isinstance(hit, Mapping) or not (
                (hit.get("current") is True and hit.get("superseded") is False) or
                (slot.get("temporal_contract") == "historical_source_v1" and
                 hit.get("source_version") and hit.get("temporal_status") == "unknown"))):
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
        if slot.get("temporal_contract") == "historical_source_v1" and hit.get("source_version") != hashlib.sha256(
                json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest():
            return None
        rows[ref] = (store_id, start, copy.deepcopy(row))
        candidates.append({"id": ref, "ref": ref, "candidate_id": ref, "exact_ref": ref,
                           "authorized": True, "scope": engine.current_session_id,
                           "session": row["session_id"], "current": hit.get("current"),
                           "superseded": hit.get("superseded"),
                           "source_version": hit.get("source_version"),
                           "temporal_status": hit.get("temporal_status"),
                           "excerpt_complete": start == 0 and end == len(content),
                           "excerpt": content[start:end], "session_id": row["session_id"],
                           "role": row.get("role"), "source": row.get("source"),
                           "timestamp": row.get("timestamp")})
    response = _request(facade, "expand_one_owned_ref", event="missing_history_slot",
                        facts={"question": question, "missing_slot": question, "candidates": candidates,
                               "visible_refs": list(visible), "recovery_budget": 1,
                               "already_visible": False,
                               "explicit_ref_available": slot.get("explicit_ref_available"),
                               "scope": engine.current_session_id,
                               "temporal_contract": slot.get("temporal_contract")},
                        completeness={"archive_complete": False}, deadline=deadline)
    if response is None:
        return None
    if response.get("relation") != "states_missing_decision":
        acknowledge(facade, response)
        return None
    selected = response.get("candidate_id")
    if not isinstance(selected, str) or selected not in rows:
        acknowledge(facade, response)
        return None
    store_id, start, before = rows[selected]
    if slot.get("temporal_contract") == "historical_source_v1" and facade.history_source_absent(
            selected, before["content"]) is not True:
        acknowledge(facade, response)
        return None
    # Revalidate live owner scope and exact source bytes after the judgment.
    if (time.monotonic() >= deadline
            or before.get("session_id") not in set(_recent_conversation_scope_session_ids(engine))
            or engine._store.get(store_id) != before):
        acknowledge(facade, response)
        return None
    with _RECOVERY_LOCK:
        recovered = getattr(engine, "_supervision_recovered_slots", set())
        if key in recovered or len(recovered) >= 32:
            acknowledge(facade, response)
            return None
        engine._supervision_recovered_slots = recovered | {key}
    try:
        from contextlib import nullcontext
        # Legacy non-native adapters have no host selection capability. Native
        # selections MUST acquire the host's revision/registration read fence.
        native = response.get("consumption") == "supervision.owner-consumption.v1"
        fence = facade.begin_history_expansion(response) if native else nullcontext(True)
        with fence as admitted:
            if not admitted:
                acknowledge(facade, response)
                return None
            if (time.monotonic() >= deadline or
                    (slot.get("temporal_contract") == "historical_source_v1" and
                     facade.history_source_absent(selected, before["content"]) is not True)):
                acknowledge(facade, response)
                return None
            result = json.loads(lcm_expand({"store_id": store_id, "content_offset": start,
                                           "max_tokens": 600, "include_exact_ref": True}, engine=engine))
    except Exception:
        acknowledge(facade, response)
        return None
    if (time.monotonic() >= deadline or result.get("session_id") != before["session_id"]
            or result.get("role") != before.get("role") or "error" in result
            or result.get("content") != before.get("content")
            or result.get("exact_ref") != selected
            or result.get("content_offset") != start
            or result.get("content_returned_chars") != len(before.get("content", ""))
            or result.get("conversation_id", "") != (before.get("conversation_id") or "")
            or result.get("source", "") != (before.get("source") or "")
            or engine._store.get(store_id) != before
            or before["session_id"] not in set(_recent_conversation_scope_session_ids(engine))):
        acknowledge(facade, response)
        return None
    # The ordinary exact-history reader owns bytes, role, lineage and offsets.
    return result if acknowledge(facade, response, result) else None

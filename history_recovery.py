"""Opt-in missing historical decision on the existing scoped grep tool.

The tool's missing_decision is an explicit request, never a source-derived grant.
Only that invocation's returned message IDs can supply candidates. Exact native
rows establish source currency, NOT decision currency or absence of supersession.
"""
from __future__ import annotations
import hashlib
import json
import time
from .decision_adapter import supports, recover_missing_history


def recover_from_grep(engine, args, response, *, deadline):
    facade = getattr(engine, "supervision", None)
    question = args.get("missing_decision")
    if (not isinstance(question, str) or not 0 < len(question) <= 1200 or
            args.get("session_scope", "current") != "current" or
            args.get("content_scope", "history") != "history" or
            not supports(facade, "expand_one_owned_ref") or "error" in response):
        return None
    absent = getattr(facade, "history_source_absent", None)
    if not callable(absent):
        return None
    hits = []
    for hit in response.get("results", ())[:8]:
        sid = hit.get("store_id")
        if type(sid) is not int:
            continue
        row = engine._store.get(sid)
        if not row or row.get("session_id") != engine.current_session_id or row.get("role") not in {"user", "assistant", "developer", "system", "tool"}:
            continue
        text = row.get("content")
        if not isinstance(text, str) or not 0 < len(text) <= 1200:
            continue
        ref = f"lcm:{sid}:0-{len(text)}"
        if absent(ref, text) is not True:
            continue
        hits.append({"exact_ref": ref, "excerpt": text, "temporal_status": "unknown",
                     "source_version": hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
    if not hits or time.monotonic() >= deadline:
        return None
    slot_id = hashlib.sha256((engine.current_session_id + "\0" + question).encode()).hexdigest()
    return recover_missing_history(engine, {"slot_id": slot_id, "question": question,
        "expansion_budget": 1, "explicit_ref_available": False, "visible_refs": [],
        "temporal_contract": "historical_source_v1", "hits": hits}, deadline=deadline)

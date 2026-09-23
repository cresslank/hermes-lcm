"""Recall delivery shaping; shared unchanged validators for baseline and selected views."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any
from .reasoning import resolve_occurrence_time


def shape_recall_response(*, ordered, rerank_status, engine, query, limit, scope_bias, include, detail, delta_requested, reference_strict, arm_order, arm_weights, coverage, embedding_query_metrics, degraded_reasons, timed_out, requested_limit, seen_refs, include_occurrence_time, summary_leads):
    from .tools import (
        _LCM_RECALL_ANSWER_READY_CONTENT_CHARS,
        _LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT,
        _LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT,
        _LCM_RECALL_LIMIT_CAP,
        _LCM_RECALL_RESPONSE_CHAR_CAP,
        _LCM_RECALL_SNIPPET_CHARS,
        _LcmRecallStrictSelector,
        _hit_identity,
        _lcm_recall_answer_ready_content,
        _lcm_recall_diverse_entries,
        _lcm_recall_exact_ref
    )
    # -- Response shaping (char-capped). The default snippets path retains the
    # historical order and serialized response exactly. answer_ready applies
    # stable post-rank diversity before bounded exact-ref hydration.
    diversity_dropped = 0
    if detail == "answer_ready":
        # Take only what the response can hold. Delta used to select the whole
        # 25-candidate cap up front and filter afterwards, which walked the
        # ranking to exhaustion while already-seen entries held session quota --
        # the refill then had nothing left to resume into. Selecting a wave at a
        # time lets a released slot be reused by the next wave.
        selection_limit = limit
        expanded_limit = (
            _LCM_RECALL_LIMIT_CAP
            if delta_requested
            else _LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT
        )
        if reference_strict:
            strict_selector = _LcmRecallStrictSelector(
                ordered,
                engine=engine,
                per_session_limit=_LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT,
                expanded_limit=expanded_limit,
            )
            selected_entries = strict_selector.take(selection_limit)
            strict_rows = strict_selector.rows
        else:
            selected_entries, diversity_dropped = _lcm_recall_diverse_entries(
                ordered,
                limit=selection_limit,
                per_session_limit=_LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT,
            )
            strict_rows = None
        answer_ready_content = _lcm_recall_answer_ready_content(
            engine,
            selected_entries,
            query=query,
            expanded_limit=expanded_limit,
            rows_by_id=strict_rows,
        )
        if delta_requested:
            def _novel(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
                """Keep the entries carrying a reference the caller lacks.

                A discarded entry hands its session slot back: it is not part of
                the response, so it must not count against the density budget
                that decides which novel rows can still be delivered.
                """
                kept: list[dict[str, Any]] = []
                for entry in entries:
                    exact_ref = (
                        None
                        if entry["hit"].get("kind") == "summary"
                        else _lcm_recall_exact_ref(
                            entry["hit"],
                            answer_ready_content.get(_hit_identity(entry["hit"])),
                        )
                    )
                    if exact_ref is not None and exact_ref not in seen_refs:
                        kept.append(entry)
                    elif reference_strict:
                        strict_selector.release(entry)
                return kept

            selected_entries = _novel(selected_entries)
            # Delta shaping discards entries the caller has already seen, so it
            # too must be able to draw on the ranked tail -- otherwise the mode
            # silently returns short while valid candidates remain, which is the
            # very underfill the resumable walk exists to prevent.
            while (
                reference_strict
                and len(selected_entries) < limit
                and not strict_selector.exhausted()
            ):
                more = strict_selector.take(limit)
                if not more:
                    break
                answer_ready_content.update(
                    _lcm_recall_answer_ready_content(
                        engine,
                        more,
                        query=query,
                        expanded_limit=len(more),
                        rows_by_id=strict_selector.rows,
                    )
                )
                selected_entries.extend(_novel(more))
            selected_entries = selected_entries[:limit]
    else:
        selected_entries = ordered
        diversity_dropped = 0
        answer_ready_content = {}
    hits_out: list[dict[str, Any]] = []
    response_chars = 0
    response_cap_truncated = False
    unreferenced_omitted = 0
    for entry in selected_entries:
        hit = entry["hit"]
        arms = sorted({arm_order[index] for index in entry["ranks"].keys()})
        item: dict[str, Any] = {
            "kind": hit.get("kind"),
            "session_id": hit.get("session_id"),
            "timestamp": hit.get("timestamp") or 0,
            "snippet": (hit.get("snippet") or "")[:_LCM_RECALL_SNIPPET_CHARS],
            "score": round(float(entry["_final_score"]), 6),
            "expand_hint": hit.get("expand_hint"),
            "from_current_session": bool(hit.get("from_current_session")),
            "arms": arms,
        }
        if hit.get("kind") == "summary":
            item["node_id"] = hit.get("node_id")
            if hit.get("store_id") is not None:
                item["store_id"] = hit.get("store_id")
        else:
            item["store_id"] = hit.get("store_id")
            if hit.get("chunk_span"):
                item["chunk_span"] = hit["chunk_span"]
        if detail == "answer_ready":
            item["role"] = hit.get("role")
            item["source"] = hit.get("source") or (
                "summary" if hit.get("kind") == "summary" else ""
            )
            hydrated = answer_ready_content.get(_hit_identity(hit))
            if hydrated is not None:
                item.update(hydrated)
            if delta_requested:
                exact_ref = _lcm_recall_exact_ref(hit, hydrated)
                if exact_ref is not None:
                    item["exact_ref"] = exact_ref
            if include_occurrence_time and hit.get("kind") != "summary":
                session_dates = getattr(engine, "_session_occurrence_dates", {}) or {}
                source_row = engine._store.get(int(hit.get("store_id") or 0))
                source_row = source_row or {}
                source_observed_at = source_row.get("observed_at")
                session_date = session_dates.get(str(hit.get("session_id")))
                if session_date is None and source_observed_at is not None:
                    try:
                        session_date = datetime.fromtimestamp(
                            float(source_observed_at), tz=timezone.utc
                        ).date().isoformat()
                    except (TypeError, ValueError, OverflowError, OSError):
                        session_date = None
                occurrence = resolve_occurrence_time(
                    (hydrated or {}).get("content") or hit.get("snippet") or "",
                    observed_at=source_observed_at or 0,
                    session_date=session_date,
                )
                occurrence["stored_at"] = source_row.get("ingested_at") or source_row.get("timestamp")
                item["occurrence_time"] = occurrence
                item["observation_time"] = {
                    "observed_at": occurrence.get("observed_at") or None,
                    "ingested_at": source_row.get("ingested_at") or source_row.get("timestamp"),
                    "source": (
                        "benchmark_session_date"
                        if str(hit.get("session_id")) in session_dates
                        else "host_message_timestamp"
                        if source_observed_at is not None
                        else "ingest_fallback"
                    ),
                }
        if reference_strict:
            # Selection already proved this candidate against its row, so there
            # is nothing left to re-check here -- only the proven span to
            # PUBLISH, so a consumer never has to guess an offset. (__init__.py's
            # _answer_ready_baseline substitutes 0 when the field is absent.)
            if hydrated is not None:
                span = (
                    int(hydrated["content_offset"]),
                    int(hydrated["content_returned_chars"]),
                )
            else:
                span = entry.get("_strict_span")
            if span is None:
                unreferenced_omitted += 1
                continue
            item["content_offset"], item["content_returned_chars"] = span
        item_chars = len(json.dumps(item, ensure_ascii=False))
        if hits_out and response_chars + item_chars > _LCM_RECALL_RESPONSE_CHAR_CAP:
            response_cap_truncated = True
            break
        response_chars += item_chars
        if reference_strict:
            strict_selector.deliver(entry)
        hits_out.append(item)
        if len(hits_out) >= limit:
            break

    degraded = bool(degraded_reasons)
    response: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "scope_bias": scope_bias,
        "include": include,
        "total_results": len(hits_out),
        "hits": hits_out,
        "provenance": {
            "arms_run": arm_order,
            "arm_weights": {name: arm_weights[i] for i, name in enumerate(arm_order)},
            "coverage": coverage,
            "rerank": rerank_status,
            "ordering": (
                "rrf-fusion -> scope/recency prior -> rerank reorder (top window); "
                "the reported score is the scope/recency-adjusted RRF score, and "
                "rerank (when applied) only permutes the top window without "
                "replacing that score"
            ),
        },
        "metrics": {
            "embedding_query_calls": len(embedding_query_metrics),
            "embedding_query_tokens": sum(
                int(item["usage_tokens"] or 0) for item in embedding_query_metrics
            ),
            "embedding_query_tokens_complete": all(
                item["usage_tokens"] is not None for item in embedding_query_metrics
            ),
            "embedding_queries": embedding_query_metrics,
        },
        "degraded": degraded,
    }
    if degraded:
        response["degraded_reason"] = "; ".join(dict.fromkeys(degraded_reasons))
    if timed_out:
        response["timeout"] = True
    if requested_limit > _LCM_RECALL_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    if detail == "answer_ready":
        expansion = {
            "expanded_hit_count": sum("content" in hit for hit in hits_out),
            "expanded_hit_limit": _LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT,
            "per_session_limit": _LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT,
            "diversity_dropped_count": diversity_dropped,
            "per_hit_char_cap": _LCM_RECALL_ANSWER_READY_CONTENT_CHARS,
            "snippet_char_cap": _LCM_RECALL_SNIPPET_CHARS,
            "response_char_cap": _LCM_RECALL_RESPONSE_CHAR_CAP,
            "response_policy": (
                "rank-preserving session diversity, then exact-ref hydration; "
                "whole hits only when enforcing the response cap"
            ),
            "hydration_policy": "bounded exact reads only; no additional retrieval search",
            "response_truncated": response_cap_truncated,
        }
        if reference_strict:
            expansion["reference_strict"] = True
            expansion["diversity_dropped_count"] = strict_selector.diversity_dropped
            expansion["unreferenced_dropped_count"] = (
                strict_selector.unreferenced_dropped
            )
            expansion["unreferenced_omitted_count"] = unreferenced_omitted
            expansion["summary_leads"] = summary_leads
            expansion["reference_policy"] = (
                "every delivered hit publishes the (store_id, content_offset, "
                "content_returned_chars) span its text occupies in the current "
                "row; a candidate that fails that check is replaced by the "
                "next-ranked citable one. Summary nodes are never delivered as "
                "evidence -- their relevance reaches the ranking through the "
                "source messages beneath them, and the nodes themselves come "
                "back here as non-evidence drill-down leads"
            )
        response["detail"] = detail
        response["provenance"]["detail"] = detail
        response["provenance"]["answer_ready"] = expansion
        if delta_requested:
            novel_refs = [hit["exact_ref"] for hit in hits_out if hit.get("exact_ref")]
            response["delta"] = {
                "protocol": "exact-ref-delta-v1",
                "seen_ref_count": len(seen_refs),
                "novel_ref_count": len(novel_refs),
                "novel_refs": novel_refs,
                "progress": bool(novel_refs),
                "termination_reason": None if novel_refs else "no_novel_exact_ref",
            }
        if include_occurrence_time:
            response["provenance"]["occurrence_time"] = {
                "policy_version": "occurrence-time-v1",
                "anchor_source": "engine session metadata when available",
                "observation_is_not_occurrence": True,
            }

        encoded = json.dumps(response, ensure_ascii=False)
        if len(encoded) > _LCM_RECALL_RESPONSE_CHAR_CAP:
            original_query = response["query"]
            response["query"] = original_query[:4_096]
            expansion["query_truncated"] = len(response["query"]) < len(original_query)
            encoded = json.dumps(response, ensure_ascii=False)
        while len(encoded) > _LCM_RECALL_RESPONSE_CHAR_CAP and (
            response["hits"] or expansion.get("summary_leads")
        ):
            if response["hits"]:
                response["hits"].pop()
            else:
                expansion["summary_leads"].pop()
            response["total_results"] = len(response["hits"])
            expansion["response_truncated"] = True
            expansion["expanded_hit_count"] = sum(
                "content" in hit for hit in response["hits"]
            )
            encoded = json.dumps(response, ensure_ascii=False)
        if delta_requested:
            novel_refs = [
                hit["exact_ref"]
                for hit in response["hits"]
                if hit.get("exact_ref")
            ]
            response["delta"].update(
                {
                    "novel_ref_count": len(novel_refs),
                    "novel_refs": novel_refs,
                    "progress": bool(novel_refs),
                    "termination_reason": (
                        None if novel_refs else "no_novel_exact_ref"
                    ),
                }
            )
            encoded = json.dumps(response, ensure_ascii=False)
        return encoded
    return json.dumps(response)

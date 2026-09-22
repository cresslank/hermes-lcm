"""Closed original-source grammar. Identity/integrity are not truth or authority.

No extraction, aliases, conversion, observation-time fallback, or store access.
The caller must supply the already hydrated WHOLE native message row.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re

SCHEMA = "lcm.literal-record.v1"
MAX_BYTES = 2400
# Deliberately finite unit identities; no spelling normalization or conversions.
UNITS = frozenset({"1", "count", "m", "s", "kg", "K", "USD", "EUR", "percent"})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})\Z")
ROW_FIELDS = ("store_id", "session_id", "source", "role", "content", "tool_call_id",
              "tool_calls", "tool_name", "timestamp", "conversation_id", "ingested_at",
              "observed_at", "observed_at_source")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(token):
    """Only admit tokens faithfully expressible by the existing JSON encoder.

    Decimal is an input oracle, not a new coordinate type. Check the original
    bound before conversion, then compare numerical meaning after canonical
    encoding. This preserves ordinary 0.1 while rejecting underflow/rounding.
    copy_abs avoids Decimal's ambient arithmetic precision rounding the bound.
    """
    try:
        original = Decimal(token)
        if not original.is_finite() or original.copy_abs() > 2**53 - 1:
            raise ValueError("literal_value_unknown")
        value = float(token)
        if not math.isfinite(value) or Decimal(canonical(value)) != original:
            raise ValueError("literal_value_unrepresentable")
        return value
    except InvalidOperation as exc:
        raise ValueError("literal_value_unrepresentable") from exc


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys.split()):
        raise ValueError("literal_shape")


def _identity(value):
    _keys(value, "namespace id")
    if any(type(v) is not str or not _ID.fullmatch(v) for v in value.values()):
        raise ValueError("literal_identity")


def _scalar(value):
    if type(value) is bool:
        return
    if type(value) is str and 0 < len(value) <= 256 and not any(ord(c) < 32 for c in value):
        return
    if type(value) is int and abs(value) <= 2**53 - 1:
        return
    if type(value) is float and math.isfinite(value) and abs(value) <= 2**53 - 1:
        return
    raise ValueError("literal_value_unknown")


@dataclass(frozen=True)
class LiteralRecord:
    schema_id: str
    exact_ref: str
    row_hash: str
    record_span: tuple[int, int]  # Unicode character offsets, like LCM exact refs
    source_bytes: bytes         # exact stored UTF-8; never canonicalized
    coordinate_pins: tuple[tuple[str, str], ...]
    source_attribution: tuple[tuple[str, str], ...]


def validate_row(row, start, end):
    """Return immutable coordinates from an original row, or abstain entirely."""
    try:
        content = row["content"]
        if (type(content) is not str or type(start) is not int or type(end) is not int
                or start != 0 or end != len(content) or row.get("role") not in {"user", "tool"}
                or type(row.get("store_id")) is not int or row["store_id"] <= 0
                or not row.get("session_id")):
            return None
        source_bytes = content.encode("utf-8", errors="strict")
        if not 0 < len(source_bytes) <= MAX_BYTES:
            return None
        record = json.loads(content, object_pairs_hook=_object, parse_float=_number,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        _keys(record, "schema entity predicate time scope quantity value polarity modality")
        if record["schema"] != SCHEMA:
            return None
        _identity(record["entity"])
        _identity(record["predicate"])
        time = record["time"]
        _keys(time, "basis start end")
        if time["basis"] == "atemporal":
            if time["start"] is not None or time["end"] is not None:
                return None
        elif time["basis"] == "event_interval":
            for value in (time["start"], time["end"]):
                if type(value) is not str or not _TIME.fullmatch(value) or value.endswith("-00:00"):
                    return None
                # Reject offsets outside ISO hours/minutes (fromisoformat normalizes minutes).
                if value[-1] != "Z" and (int(value[-5:-3]) > 23 or int(value[-2:]) > 59):
                    return None
            if datetime.fromisoformat(time["start"].replace("Z", "+00:00")) >= datetime.fromisoformat(time["end"].replace("Z", "+00:00")):
                return None
        else:
            return None
        scope = record["scope"]
        _keys(scope, "namespace id conditions")
        _identity({k: scope[k] for k in ("namespace", "id")})
        conditions = scope["conditions"]
        if type(conditions) is not list or len(conditions) > 8:
            return None
        # Conditions are explicit equality constraints; no free-prose qualifier interpretation.
        seen = set()
        for condition in conditions:
            _keys(condition, "namespace id value")
            _identity({k: condition[k] for k in ("namespace", "id")})
            _scalar(condition["value"])
            key = (condition["namespace"], condition["id"])
            if key in seen:
                return None
            seen.add(key)
        quantity = record["quantity"]
        _keys(quantity, "kind unit")
        _scalar(record["value"])
        if quantity["kind"] == "quantity":
            if type(quantity["unit"]) is not str or quantity["unit"] not in UNITS or type(record["value"]) not in (int, float):
                return None
        elif quantity["kind"] == "non_quantity":
            if quantity["unit"] is not None:
                return None
        else:
            return None
        if record["polarity"] not in ("positive", "negative") or record["modality"] not in ("asserted", "possible", "necessary"):
            return None
        # Pin all native attribution/version fields, including observed time, without
        # using any of them to manufacture event semantics. Mutable pin flags excluded.
        metadata = {k: row.get(k) for k in ROW_FIELDS if k != "content"}
        if any(type(v) is str and len(v) > 2400 for v in metadata.values()):
            return None
        if len(canonical(metadata).encode()) > 4096:
            return None
        row_hash = hashlib.sha256(canonical({k: row.get(k) for k in ROW_FIELDS}).encode()).hexdigest()
        attribution = tuple((k, canonical(row.get(k))) for k in ROW_FIELDS if k != "content")
        return LiteralRecord(SCHEMA, f"lcm:{row['store_id']}:0-{end}", row_hash,
                             (0, end), source_bytes,
                             tuple((k, canonical(record[k])) for k in
                                   ("entity", "predicate", "time", "scope", "quantity", "value", "polarity", "modality")),
                             attribution)
    except (KeyError, TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        return None


def observe_hydrated(engine, row, start, end):
    """An invocation-local observer, never a new lookup or a public fact setter."""
    from .literal_source import observe_hydrated as observe
    observe(engine, row, start, end)

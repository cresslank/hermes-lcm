"""Storage-boundary protection for payloads that should not live inline in SQLite.

Hermes core may hand LCM messages that already contain inline media/base64
payloads. LCM remains lossless by externalizing those payload strings and
storing compact placeholders in ``messages.content`` / ``messages.tool_calls``.
This avoids duplicating large/binary-ish payloads into SQLite rows, FTS shadow
structures, WAL files, and backups while keeping recovery available through LCM
externalized-payload tools.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .externalize import (
    externalize_ingest_payload,
    extract_externalized_ref,
    extract_externalized_refs,
    find_externalized_payload_for_message,
    get_large_output_storage_dir,
    is_externalized_placeholder,
    load_externalized_payload,
    maybe_externalize_payload,
)
from .message_content import normalize_content_value

logger = logging.getLogger(__name__)

_MEDIA_TYPE_HINTS = ("image", "audio", "video")
_MEDIA_VALUE_KEYS = (
    "image_url",
    "input_image",
    "output_image",
    "audio_url",
    "video_url",
    "image",
    "audio",
    "video",
)


def _contains_media_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_DATA_URI_BASE64_RE.search(value))
    if isinstance(value, list):
        return any(_contains_media_payload(item) for item in value)
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").lower()
        if any(hint in block_type for hint in _MEDIA_TYPE_HINTS):
            return True
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in _MEDIA_VALUE_KEYS:
                return True
            if _contains_media_payload(nested):
                return True
    return False


def _externalization_kind_for_message(message: Dict[str, Any]) -> str:
    role = str(message.get("role") or "unknown")
    if role == "tool":
        return "tool_result"
    if _contains_media_payload(message.get("content")):
        return "media_payload"
    return "raw_payload"


# Any data URI base64 payload, not just image/audio/video. Keep the trailing
# payload alphabet conservative so we do not slurp surrounding JSON/markdown.
# Raw scans can see JSON-escaped slashes before decoding, including both `\/`
# and unicode escapes such as `\u002f` in duplicate-key argument strings.
_JSON_ESCAPED_SLASH_RE = r"(?:/|\\/|\\u002[fF])"
_DATA_URI_BASE64_RE = re.compile(
    rf"data:(?:[A-Za-z0-9.+-]|{_JSON_ESCAPED_SLASH_RE})*"
    rf"(?:;[A-Za-z0-9_.+%-]+=(?:[-A-Za-z0-9_.+%]|{_JSON_ESCAPED_SLASH_RE})*)*"
    rf";base64,(?:[A-Za-z0-9+=]|{_JSON_ESCAPED_SLASH_RE}){{256,}}(?=$|[^A-Za-z0-9+/=])",
    re.IGNORECASE,
)

_BASE64_RUN_RE = re.compile(r"(?<![A-Za-z0-9+/=_-])([A-Za-z0-9+/=_-]{4096,})(?![A-Za-z0-9+/=_-])")
# Line-wrapped base64 (MIME 76 / PEM 64 chars per line) never forms a single
# 4096-char contiguous run, so _BASE64_RUN_RE misses it entirely. Match a block
# of consecutive base64-alphabet lines; looks_like_long_base64 makes the final
# call on the whitespace-compacted block.
_WRAPPED_BASE64_MIN_LINE_CHARS = 40
_WRAPPED_BASE64_MIN_TERMINAL_LINE_CHARS = 16
_BASE64_ALPHABET_RE = re.compile(r"^[A-Za-z0-9+/=_\s-]+$")
_BASE64_LINE_ALPHABET_RE = re.compile(r"^[A-Za-z0-9+/=_-]+$")
_PRIVATE_KEY_BEGIN_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE)
_PRIVATE_KEY_END_RE = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE)
_EXTERNALIZED_PLACEHOLDER_PREFIX = "[Externalized LCM ingest payload:"
_QUARANTINED_ASSISTANT_KIND = "quarantined_assistant_output"
_QUARANTINED_ASSISTANT_REASON = "high_repetition"
_QUARANTINED_ASSISTANT_MIN_CHARS = 65_536
_QUARANTINED_ASSISTANT_MIN_TOKENS = 1_000
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_REPETITION_SEGMENT_SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
_HEARTBEAT_NOISE_RE = re.compile(
    r"^(?:still\s+working|working\s+on\s+it|processing|checking|one\s+moment|ping|heartbeat|no\s+update)(?:[.!…\s-]*)$",
    re.IGNORECASE,
)
_HEARTBEAT_NOISE_MAX_CHARS = 256
_GENERIC_BASE64_MIN_CHARS = 4096
_INGEST_PLACEHOLDER_RE = re.compile(r"\[Externalized LCM ingest payload:.*?;\s*ref=([^;\]\s]+)\]")
_EXTERNALIZED_PAYLOAD_PLACEHOLDER_RE = re.compile(
    r"\[(?:Externalized|GC'd externalized) (?:tool output|payload):.*?;\s*ref=([^;\]\s]+)\]"
)
_PERSISTED_OUTPUT_TAG = "<persisted-output>"
_PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
_PERSISTED_OUTPUT_SAVED_TO_RE = re.compile(r"^Full output saved to:\s*(?P<path>.+?)\s*$", re.MULTILINE)
_PERSISTED_OUTPUT_PREVIEW_RE = re.compile(
    r"^Preview \(first \d+ chars\):\s*\r?\n(?P<preview>.*?)\r?\n</persisted-output>\s*$",
    re.MULTILINE | re.DOTALL,
)
_PERSISTED_OUTPUT_CHAR_COUNT_RE = re.compile(r"too large\s*\((?P<count>[\d,]+)\s+characters\b", re.IGNORECASE)
_PERSISTED_OUTPUT_INLINE_PREVIEW_SHA256_RE = re.compile(
    r"\r?\n\[LCM persisted-output marker identity: preview_sha256=(?P<sha256>[0-9a-f]{64})\]"
    r"(?:\r?\n\[LCM persisted-output file generation: size=\d+; mtime_ns=\d+; ctime_ns=\d+\])?"
    r"\r?\n</persisted-output>\s*$"
)
_PERSISTED_OUTPUT_INLINE_GENERATION_RE = re.compile(
    r"\r?\n\[LCM persisted-output file generation: size=(?P<size>\d+); mtime_ns=(?P<mtime_ns>\d+); ctime_ns=(?P<ctime_ns>\d+)\]\r?\n</persisted-output>\s*$"
)
_PERSISTED_OUTPUT_INLINE_METADATA_RE = re.compile(
    r"\r?\n\[LCM persisted-output (?:file generation|marker identity):[^\r\n]*\]\s*$"
)
_UNRECOVERABLE_TRUNCATION_RE = re.compile(
    r"\[Truncated:\s*tool response was [\d,]+ chars\.\s*Full output could not be saved to sandbox\.\]",
    re.IGNORECASE,
)
_HERMES_RESULTS_DIRNAME = "hermes-results"
_MAX_RECOVERED_PERSISTED_OUTPUT_BYTES = 64 * 1024 * 1024
_SENSITIVE_PLACEHOLDER_PREFIX = "[LCM sensitive redaction:"
_SENSITIVE_PATTERN_CATALOG: dict[str, re.Pattern[str]] = {
    "api_key": re.compile(
        r"(?P<prefix>(?:\\?[\"']?)\b(?:api[_-]?key|api[_-]?token|access[_-]?token|secret[_-]?key|client[_-]?secret)\b\s*(?:\\?[\"']?)\s*[:=]\s*(?:\\?[\"']?))"
        r"(?P<secret>[A-Za-z0-9._~+/=-]{12,})"
        r"(?P<suffix>\\?[\"']?)",
        re.IGNORECASE,
    ),
    "bearer_token": re.compile(
        r"(?P<prefix>\bBearer\s+)"
        r"(?P<secret>[A-Za-z0-9._~+/=-]{12,})",
        re.IGNORECASE,
    ),
    "password_assignment": re.compile(
        r"(?P<prefix>\b(?:password|passwd|pwd|passphrase)\b\s*[\"']?\s*[:=]\s*)"
        r"(?:(?P<quote>[\"'])(?P<secret_quoted>[^\r\n\]\}]{6,}?)(?P=quote)|"
        r"(?P<secret_unquoted>[^\s,\"'\]}]{6,}))",
        re.IGNORECASE,
    ),
    "private_key": re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.IGNORECASE | re.DOTALL,
    ),
}

# Sensitive redaction runs synchronously in the ingest path. The private_key
# pattern (lazy `.*?` under DOTALL) rescans to end-of-string for every unmatched
# BEGIN header, so a multi-MB payload with many headers and no END is O(n^2) and
# can block a turn for minutes. Guard it: prefer the optional `regex` engine with
# a match timeout (fail-open on timeout), and when `regex` is unavailable bound
# the input length the stdlib DOTALL pattern is applied to.
try:  # pragma: no cover - exercised when the optional dependency is absent
    import regex as _regex_engine
except Exception:  # pragma: no cover - keep the plugin importable in minimal installs
    _regex_engine = None

_SENSITIVE_MATCH_TIMEOUT_SECONDS = 1.0
# Legitimate PEM keys are a few KB; above this a DOTALL rescan is the attack, not
# a real key, so fail-open rather than block ingest.
_SENSITIVE_STDLIB_MAX_CHARS = 262_144
_BACKTRACKING_RISKY_SENSITIVE_PATTERNS = frozenset({"private_key"})
_SENSITIVE_TIMEOUT_WARNED: set[str] = set()
_SENSITIVE_STDLIB_SKIP_WARNED: set[str] = set()
_SENSITIVE_REGEX_CATALOG: dict[str, Any] = {}


def _regex_engine_flags(re_flags: int) -> int:
    mapped = 0
    if re_flags & re.IGNORECASE:
        mapped |= _regex_engine.IGNORECASE
    if re_flags & re.DOTALL:
        mapped |= _regex_engine.DOTALL
    if re_flags & re.MULTILINE:
        mapped |= _regex_engine.MULTILINE
    if re_flags & re.VERBOSE:
        mapped |= _regex_engine.VERBOSE
    return mapped


def _regex_pattern_for(name: str) -> Any:
    """Lazily compile the timeout-capable `regex` mirror of a catalog pattern."""
    if _regex_engine is None:
        return None
    cached = _SENSITIVE_REGEX_CATALOG.get(name)
    if cached is not None:
        return cached
    stdlib_pattern = _SENSITIVE_PATTERN_CATALOG[name]
    compiled = _regex_engine.compile(
        stdlib_pattern.pattern, _regex_engine_flags(stdlib_pattern.flags)
    )
    _SENSITIVE_REGEX_CATALOG[name] = compiled
    return compiled


def _apply_sensitive_pattern(name: str, repl, text: str) -> str:
    """Substitute one sensitive pattern with a ReDoS-safe strategy.

    Fails open (leaves the span unredacted) with a one-time warning rather than
    blocking the ingest path on a pathological input.
    """
    # Only the private_key pattern (lazy `.*?` under DOTALL, which rescans to
    # end-of-string per unmatched BEGIN header) is O(n^2) and needs a guard.
    # The other patterns are character-class-bounded and linear, so they always
    # run via stdlib and never fail open - a redaction bypass under CPU load
    # would be a silent secret leak, so we restrict fail-open to the one
    # pattern that genuinely requires it.
    if name in _BACKTRACKING_RISKY_SENSITIVE_PATTERNS:
        regex_pattern = _regex_pattern_for(name)
        if regex_pattern is not None:
            try:
                return regex_pattern.sub(
                    repl, text, timeout=_SENSITIVE_MATCH_TIMEOUT_SECONDS
                )
            except TimeoutError:
                if name not in _SENSITIVE_TIMEOUT_WARNED:
                    _SENSITIVE_TIMEOUT_WARNED.add(name)
                    logger.warning(
                        "LCM sensitive redaction %r timed out after %.3gs; leaving "
                        "span unredacted for this input",
                        name,
                        _SENSITIVE_MATCH_TIMEOUT_SECONDS,
                    )
                return _redact_private_key_blocks(text)
        elif name == "private_key":
            return _redact_private_key_blocks(text)
    return _SENSITIVE_PATTERN_CATALOG[name].sub(repl, text)




def _redact_private_key_blocks(text: str) -> str:
    """Redact PEM private-key blocks with a linear scanner.

    This keeps large valid keys protected even when the optional ``regex``
    package is unavailable, without running the stdlib DOTALL private-key
    pattern over a pathological multi-MB input. Unmatched BEGIN headers are
    left intact; there is no complete key block to redact.
    """
    if "private key-----" not in text.lower():
        return text
    parts: list[str] = []
    cursor = 0
    changed = False
    while True:
        begin = _PRIVATE_KEY_BEGIN_RE.search(text, cursor)
        if begin is None:
            parts.append(text[cursor:])
            break
        end = _PRIVATE_KEY_END_RE.search(text, begin.end())
        if end is None:
            parts.append(text[cursor:])
            break
        block_end = end.end()
        secret = text[begin.start():block_end]
        parts.append(text[cursor:begin.start()])
        parts.append(_sensitive_placeholder("private_key", secret))
        cursor = block_end
        changed = True
    return "".join(parts) if changed else text


def _is_wrapped_base64_line(line: str) -> bool:
    stripped = line.strip("\r\n")
    return (
        len(stripped) >= _WRAPPED_BASE64_MIN_LINE_CHARS
        and _BASE64_LINE_ALPHABET_RE.fullmatch(stripped) is not None
    )


def _is_wrapped_base64_terminal_line(line: str) -> bool:
    stripped = line.strip("\r\n")
    return (
        _WRAPPED_BASE64_MIN_TERMINAL_LINE_CHARS
        <= len(stripped)
        < _WRAPPED_BASE64_MIN_LINE_CHARS
        and len(stripped) % 4 == 0
        and _BASE64_LINE_ALPHABET_RE.fullmatch(stripped) is not None
    )


def _looks_like_hex_hash_inventory(payload: str) -> bool:
    """Return True for newline inventories of hex digests, not base64 payloads."""
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    digest_lengths = {40, 56, 64, 96, 128}
    return all(
        len(line) in digest_lengths and re.fullmatch(r"[0-9a-fA-F]+", line) is not None
        for line in lines
    )


def _iter_wrapped_base64_blocks(text: str):
    """Yield (start, end, payload) for line-wrapped base64 blocks.

    Implemented as a line scanner instead of a wide regex so long
    base64-alphabet single lines that are not actually wrapped do not trigger
    repeated failed block matches.
    """
    offset = 0
    block_start: int | None = None
    block_parts: list[str] = []
    block_end = 0

    def finish_block():
        nonlocal block_start, block_parts, block_end
        if block_start is not None and block_parts:
            payload = "".join(block_parts)
            start, end = block_start, block_end
            block_start = None
            block_parts = []
            block_end = 0
            if not _looks_like_hex_hash_inventory(payload) and looks_like_long_base64(payload):
                return (start, end, payload)
        block_start = None
        block_parts = []
        block_end = 0
        return None

    for line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        if _is_wrapped_base64_line(line) or (
            block_start is not None
            and block_parts
            and _is_wrapped_base64_terminal_line(line)
        ):
            if block_start is None:
                block_start = line_start
            block_parts.append(line)
            block_end = offset
            continue
        block = finish_block()
        if block is not None:
            yield block
    block = finish_block()
    if block is not None:
        yield block


def _replace_wrapped_base64_blocks(text: str, replace) -> str:
    chunks: list[str] = []
    cursor = 0
    changed = False
    for start, end, payload in _iter_wrapped_base64_blocks(text):
        chunks.append(text[cursor:start])
        chunks.append(replace(payload))
        cursor = end
        changed = True
    if not changed:
        return text
    chunks.append(text[cursor:])
    return "".join(chunks)


def is_externalized_ingest_placeholder(text: str) -> bool:
    return isinstance(text, str) and bool(_INGEST_PLACEHOLDER_RE.fullmatch(text.strip()))


def _is_unrecoverable_tool_truncation_marker(text: str | None) -> bool:
    return isinstance(text, str) and bool(_UNRECOVERABLE_TRUNCATION_RE.search(text))


def _expected_persisted_output_chars(text: str | None) -> int | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_CHAR_COUNT_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group("count").replace(",", ""))
    except ValueError:
        return None


def _persisted_output_preview_prefix(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_PREVIEW_RE.search(text.strip())
    if not match:
        return None
    preview = match.group("preview")
    if preview.endswith("\r\n..."):
        preview = preview[: -len("\r\n...")]
    elif preview.endswith("\n..."):
        preview = preview[: -len("\n...")]
    return preview


def _persisted_output_preview_prefix_digest(text: str | None) -> str | None:
    preview_prefix = _persisted_output_preview_prefix(text)
    if not preview_prefix:
        return None
    return hashlib.sha256(
        preview_prefix.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _persisted_output_inline_preview_sha256(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_INLINE_PREVIEW_SHA256_RE.search(text)
    if not match:
        return None
    return match.group("sha256")


def _inline_persisted_output_generation_metadata(text: str | None) -> dict[str, int] | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_INLINE_GENERATION_RE.search(text)
    if not match:
        return None
    try:
        return {
            "size": int(match.group("size")),
            "mtime_ns": int(match.group("mtime_ns")),
            "ctime_ns": int(match.group("ctime_ns")),
        }
    except (TypeError, ValueError):
        return None


def _has_inline_persisted_output_generation_metadata(text: str | None) -> bool:
    return _inline_persisted_output_generation_metadata(text) is not None


def _persisted_output_marker_identity_digest(text: str | None) -> str | None:
    return _persisted_output_inline_preview_sha256(text) or _persisted_output_preview_prefix_digest(text)


def _has_lossy_sensitive_redaction(text: str | None) -> bool:
    if not isinstance(text, str) or _SENSITIVE_PLACEHOLDER_PREFIX not in text:
        return False
    for match in re.finditer(r"\[LCM sensitive redaction: (?P<body>[^\]]+)\]", text):
        body = match.group("body")
        fields = {
            key: value
            for key, value in re.findall(r"([A-Za-z0-9_]+)=([^;]+)", body)
        }
        if fields.get("name") == "password_assignment" and "sha256" not in fields:
            return True
    return False


def _persisted_output_saved_path(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_SAVED_TO_RE.search(text.strip())
    if not match:
        return None
    raw_path = match.group("path").strip()
    if not raw_path or "\x00" in raw_path:
        return None
    return raw_path


def _safe_temp_hermes_results_file(path: Path) -> Path | None:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        return None
    parent = path.parent
    if parent.name != _HERMES_RESULTS_DIRNAME:
        return None
    try:
        expected_parent = (Path(tempfile.gettempdir()) / _HERMES_RESULTS_DIRNAME).resolve()
        parent_is_valid_dir = parent.exists() and parent.is_dir() and not parent.is_symlink()
        if not parent_is_valid_dir or parent.resolve() != expected_parent:
            return None
        return expected_parent / path.name
    except OSError:
        return None


def _is_hermes_persisted_output_marker(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    marker = text.strip()
    return (
        marker.startswith(_PERSISTED_OUTPUT_TAG)
        and marker.endswith(_PERSISTED_OUTPUT_CLOSING_TAG)
        and _expected_persisted_output_chars(marker) is not None
        and _PERSISTED_OUTPUT_SAVED_TO_RE.search(marker) is not None
    )


def _stat_generation_metadata(stats: os.stat_result) -> dict[str, int]:
    return {
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
        "ctime_ns": int(stats.st_ctime_ns),
    }


def _read_regular_file_no_symlink(path: Path) -> tuple[str, dict[str, int]] | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    fd: int | None = None
    try:
        lstat_result = os.lstat(str(path))
        if not stat.S_ISREG(lstat_result.st_mode):
            return None
        if lstat_result.st_size > _MAX_RECOVERED_PERSISTED_OUTPUT_BYTES:
            return None
        fd = os.open(str(path), flags)
        stats_before = os.fstat(fd)
        if not stat.S_ISREG(stats_before.st_mode):
            return None
        if stats_before.st_size > _MAX_RECOVERED_PERSISTED_OUTPUT_BYTES:
            return None
        with os.fdopen(fd, "rb") as handle:
            fd = None
            raw = handle.read()
            stats_after = os.fstat(handle.fileno())
        if _stat_generation_metadata(stats_before) != _stat_generation_metadata(stats_after):
            return None
        return raw.decode("utf-8"), _stat_generation_metadata(stats_after)
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def recover_hermes_persisted_output_with_file_stat(text: str | None) -> tuple[str, dict[str, int]] | None:
    """Recover Hermes host `<persisted-output>` content when the backing file is safe.

    Recovery is intentionally conservative: the marker must include Hermes'
    character count, the file path must be an absolute basename under a
    `hermes-results` temp directory, the target must be a regular non-symlink
    file, and the recovered character count must match the marker. If any check
    fails, callers should keep the marker/preview instead of claiming lossless
    recovery from an unsafe or stale file.
    """
    if not isinstance(text, str) or not _is_hermes_persisted_output_marker(text):
        return None
    expected_chars = _expected_persisted_output_chars(text)
    if expected_chars is None:
        return None
    raw_path = _persisted_output_saved_path(text)
    if raw_path is None:
        return None
    path = Path(raw_path)
    safe_path = _safe_temp_hermes_results_file(path)
    if safe_path is None:
        return None
    recovered_with_stat = _read_regular_file_no_symlink(safe_path)
    if recovered_with_stat is None:
        return None
    recovered, file_stat = recovered_with_stat
    if len(recovered) != expected_chars:
        return None
    preview_prefix = _persisted_output_preview_prefix(text)
    if not preview_prefix or not recovered.startswith(preview_prefix):
        return None
    return recovered, file_stat


def recover_hermes_persisted_output(text: str | None) -> str | None:
    recovered_with_stat = recover_hermes_persisted_output_with_file_stat(text)
    if recovered_with_stat is None:
        return None
    recovered, _file_stat = recovered_with_stat
    return recovered


def _add_inline_persisted_output_generation_metadata(text: str, file_stat: dict[str, int] | None) -> str:
    if not file_stat or not isinstance(text, str) or "</persisted-output>" not in text:
        return text
    generation = (
        "[LCM persisted-output file generation: "
        f"size={file_stat['size']}; "
        f"mtime_ns={file_stat['mtime_ns']}; "
        f"ctime_ns={file_stat['ctime_ns']}]"
    )
    if generation in text:
        return text
    return text.replace("</persisted-output>", f"{generation}\n</persisted-output>", 1)


def _add_inline_persisted_output_identity_metadata(text: str, preview_sha256: str | None) -> str:
    if (
        not isinstance(text, str)
        or "</persisted-output>" not in text
        or not isinstance(preview_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", preview_sha256)
    ):
        return text
    if _has_lossy_sensitive_redaction(text) or _persisted_output_inline_preview_sha256(text):
        return text
    identity = f"[LCM persisted-output marker identity: preview_sha256={preview_sha256}]"
    return text.replace("</persisted-output>", f"{identity}\n</persisted-output>", 1)


def contains_data_uri_base64(text: str) -> bool:
    return isinstance(text, str) and bool(_DATA_URI_BASE64_RE.search(text))


def contains_long_base64_run(text: str, *, min_chars: int = _GENERIC_BASE64_MIN_CHARS) -> bool:
    if not isinstance(text, str) or len(text) < min_chars:
        return False
    if any(looks_like_long_base64(match.group(1), min_chars=min_chars) for match in _BASE64_RUN_RE.finditer(text)):
        return True
    # Also catch line-wrapped base64 blocks (MIME/PEM), which never form a
    # single contiguous run.
    return any(
        looks_like_long_base64(payload, min_chars=min_chars)
        for _start, _end, payload in _iter_wrapped_base64_blocks(text)
    )


def extract_ingest_externalized_refs(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    refs: list[str] = []
    for match in _INGEST_PLACEHOLDER_RE.finditer(text):
        ref = match.group(1).strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _is_basename_ref(ref: str) -> bool:
    return bool(ref) and ref.endswith(".json") and "/" not in ref and "\\" not in ref and Path(ref).name == ref


def extract_all_externalized_payload_refs(text: str) -> list[str]:
    """Return deduplicated refs from recognized externalized payload placeholders."""
    if not isinstance(text, str) or not text:
        return []
    refs: list[str] = []
    for ref in extract_ingest_externalized_refs(text) + extract_externalized_refs(text):
        if _is_basename_ref(ref) and ref not in refs:
            refs.append(ref)
    return refs


def sensitive_pattern_status(config) -> dict[str, Any]:
    """Return metadata-only status for opt-in sensitive redaction."""
    configured, active, unknown = _configured_sensitive_pattern_names(config)
    enabled = bool(getattr(config, "sensitive_patterns_enabled", False))
    return {
        "sensitive_patterns_enabled": enabled,
        "enabled": enabled,
        "sensitive_patterns": configured,
        "patterns": configured,
        "active_patterns": active if enabled else [],
        "unknown_patterns": unknown,
        "source": getattr(config, "sensitive_patterns_source", "default"),
        "placeholder_format": "[LCM sensitive redaction: name=<pattern>; chars=<n>; bytes=<n>; sha256=<16 for non-password>]",
        "lossless_recovery": False if enabled and active else None,
    }


def _configured_sensitive_pattern_names(config) -> tuple[list[str], list[str], list[str]]:
    raw = getattr(config, "sensitive_patterns", []) or []
    if isinstance(raw, str):
        names = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in raw if str(part).strip()]
    if not names:
        return [], [], []
    configured: list[str] = []
    active: list[str] = []
    unknown: list[str] = []
    for name in names:
        normalized = name.lower().strip()
        if normalized in {"all", "default"}:
            for catalog_name in _SENSITIVE_PATTERN_CATALOG:
                if catalog_name not in configured:
                    configured.append(catalog_name)
                if catalog_name not in active:
                    active.append(catalog_name)
            continue
        configured.append(normalized)
        if normalized in _SENSITIVE_PATTERN_CATALOG:
            if normalized not in active:
                active.append(normalized)
        elif normalized not in unknown:
            unknown.append(normalized)
    return configured, active, unknown


def _active_sensitive_pattern_names(config) -> list[str]:
    if not bool(getattr(config, "sensitive_patterns_enabled", False)):
        return []
    _configured, active, _unknown = _configured_sensitive_pattern_names(config)
    return active


def _sensitive_placeholder(pattern_name: str, secret: str) -> str:
    parts = [
        f"{_SENSITIVE_PLACEHOLDER_PREFIX} "
        f"name={_safe_placeholder_metadata(pattern_name)}; "
        f"chars={len(secret)}; bytes={len(secret.encode('utf-8', errors='surrogatepass'))}"
    ]
    if pattern_name != "password_assignment":
        digest = hashlib.sha256(secret.encode("utf-8", errors="surrogatepass")).hexdigest()[:16]
        parts.append(f"sha256={digest}")
    return "; ".join(parts) + "]"


def _redact_match(pattern_name: str, match: re.Match[str]) -> str:
    group_names = match.re.groupindex
    secret_group = None
    for candidate in ("secret", "secret_quoted", "secret_unquoted"):
        if candidate in group_names and match.groupdict().get(candidate) is not None:
            secret_group = candidate
            break
    if secret_group is None:
        return _sensitive_placeholder(pattern_name, match.group(0))
    secret = match.group(secret_group)
    relative_start = match.start(secret_group) - match.start(0)
    relative_end = match.end(secret_group) - match.start(0)
    full = match.group(0)
    return full[:relative_start] + _sensitive_placeholder(pattern_name, secret) + full[relative_end:]


def _sensitive_pattern_for_key(key: Any, active_names: list[str]) -> str | None:
    if not isinstance(key, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    compact = normalized.replace("_", "")
    if "api_key" in active_names and (
        compact in {"apikey", "apitoken", "accesstoken", "secretkey", "clientsecret"}
        or ("api" in normalized and "key" in normalized)
        or ("access" in normalized and "token" in normalized)
        or ("secret" in normalized and "key" in normalized)
    ):
        return "api_key"
    if "bearer_token" in active_names and compact in {"authorization", "authtoken", "bearertoken", "token"}:
        return "bearer_token"
    if "password_assignment" in active_names and compact in {"password", "passwd", "pwd", "passphrase"}:
        return "password_assignment"
    return None


def redact_sensitive_text(text: str, config) -> str:
    """Replace configured sensitive spans with deterministic placeholders."""
    if not isinstance(text, str) or not text:
        return text
    active_names = _active_sensitive_pattern_names(config)
    if not active_names:
        return text
    protected = text
    for name in active_names:
        protected = _apply_sensitive_pattern(
            name,
            lambda match, pattern_name=name: _redact_match(pattern_name, match),
            protected,
        )
    return protected


def _redact_entire_sensitive_string(text: str, pattern_name: str) -> str:
    if not text or _SENSITIVE_PLACEHOLDER_PREFIX in text:
        return text
    return _sensitive_placeholder(pattern_name, text)


def redact_sensitive_value(value: Any, config, *, parse_json_strings: bool = False) -> Any:
    """Recursively redact configured sensitive values without externalizing data."""
    active_names = _active_sensitive_pattern_names(config)
    if not active_names:
        return value
    if isinstance(value, dict):
        protected: dict[Any, Any] = {}
        for key, val in value.items():
            protected_key = redact_sensitive_text(key, config) if isinstance(key, str) else key
            key_pattern = _sensitive_pattern_for_key(key, active_names)
            if key_pattern and isinstance(val, str):
                text_redacted = redact_sensitive_text(val, config)
                if text_redacted == val:
                    text_redacted = _redact_entire_sensitive_string(val, key_pattern)
                protected[protected_key] = text_redacted
            else:
                protected[protected_key] = redact_sensitive_value(
                    val,
                    config,
                    parse_json_strings=parse_json_strings,
                )
        return protected
    if isinstance(value, list):
        return [redact_sensitive_value(item, config, parse_json_strings=parse_json_strings) for item in value]
    if not isinstance(value, str):
        return value
    if parse_json_strings:
        parsed = _maybe_parse_json_string(value)
        if parsed is not None and not _json_has_duplicate_object_keys(value):
            protected = redact_sensitive_value(parsed, config, parse_json_strings=True)
            if protected != parsed:
                return json.dumps(protected, ensure_ascii=False, separators=(",", ":"))
    return redact_sensitive_text(value, config)


def _safe_placeholder_metadata(value: Any) -> str:
    text = str(value or "?")
    safe = re.sub(r"[^A-Za-z0-9_.:/-]+", "-", text).strip("-")
    return (safe or "?")[:120]


def _normalized_repetition_segments(text: str) -> list[str]:
    segments = []
    for segment in _REPETITION_SEGMENT_SPLIT_RE.split(text):
        normalized = re.sub(r"\s+", " ", segment.strip().lower())
        if len(normalized) >= 32:
            segments.append(normalized)
    return segments


def heartbeat_noise_reason(role: str, text: str) -> str | None:
    """Return a read-only doctor category for short heartbeat/progress noise.

    This intentionally does not drive ingest protection or cleanup. It only
    surfaces metadata-only candidates for operator review.
    """
    role = str(role or "")
    if role not in {"assistant", "tool", "system"}:
        return None
    if not isinstance(text, str):
        return None
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized or len(normalized) > _HEARTBEAT_NOISE_MAX_CHARS:
        return None
    if _HEARTBEAT_NOISE_RE.match(normalized):
        return "heartbeat_progress"
    return None


def assistant_output_quarantine_reason(text: str) -> str | None:
    """Return a quarantine reason for obviously broken assistant output.

    The gate is intentionally conservative: content must be very large and show
    both low token novelty and repeated sentence/line segments. Long diverse
    reports and code with varied identifiers should stay inline.
    """
    if not isinstance(text, str) or len(text) < _QUARANTINED_ASSISTANT_MIN_CHARS:
        return None

    normalized = re.sub(r"\s+", " ", text.strip().lower())
    tokens = _WORD_TOKEN_RE.findall(normalized)
    if len(tokens) < _QUARANTINED_ASSISTANT_MIN_TOKENS:
        if len(normalized) >= _QUARANTINED_ASSISTANT_MIN_CHARS and len(set(normalized)) <= 12:
            return _QUARANTINED_ASSISTANT_REASON
        return None

    token_counts = Counter(tokens)
    unique_token_ratio = len(token_counts) / max(1, len(tokens))
    top_token_ratio = token_counts.most_common(1)[0][1] / max(1, len(tokens))

    segments = _normalized_repetition_segments(text)
    top_segment_ratio = 0.0
    duplicate_segment_ratio = 0.0
    if len(segments) >= 20:
        segment_counts = Counter(segments)
        top_segment_ratio = segment_counts.most_common(1)[0][1] / len(segments)
        duplicate_segment_ratio = 1.0 - (len(segment_counts) / len(segments))

    if unique_token_ratio <= 0.03 and (
        top_segment_ratio >= 0.10
        or duplicate_segment_ratio >= 0.50
        or top_token_ratio >= 0.08
    ):
        return _QUARANTINED_ASSISTANT_REASON

    # Covers degenerate long loops with little punctuation/newline structure.
    if unique_token_ratio <= 0.015 and len(set(normalized)) <= 64:
        return _QUARANTINED_ASSISTANT_REASON

    return None


def _quarantined_assistant_placeholder(summary: Dict[str, Any], *, reason: str) -> str:
    return (
        "[Externalized LCM ingest payload: assistant output quarantined; "
        f"kind={_safe_placeholder_metadata(summary.get('kind') or _QUARANTINED_ASSISTANT_KIND)}; "
        f"reason={_safe_placeholder_metadata(reason)}; "
        f"field={_safe_placeholder_metadata(summary.get('field_path') or 'content')}; "
        f"chars={summary.get('content_chars', 0)}; bytes={summary.get('content_bytes', 0)}; "
        f"ref={summary.get('ref', '')}]"
    )


def _volatile_quarantined_assistant_placeholder(content: str, *, reason: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return (
        "[LCM active replay placeholder: assistant output quarantined; "
        f"kind={_QUARANTINED_ASSISTANT_KIND}; "
        f"reason={_safe_placeholder_metadata(reason)}; "
        "scope=ignored_message_pattern; field=content; "
        f"chars={len(content)}; bytes={len(content.encode('utf-8'))}; "
        f"sha256={digest}]"
    )


def _externalize_quarantined_assistant_output(
    content: str,
    *,
    role: str,
    session_id: str,
    config,
    hermes_home: str,
    reason: str,
) -> str | None:
    existing = find_externalized_payload_for_message(
        content,
        session_id=session_id,
        kind=_QUARANTINED_ASSISTANT_KIND,
        role=role,
        config=config,
        hermes_home=hermes_home,
    )
    if existing is not None:
        return _quarantined_assistant_placeholder(existing, reason=reason)

    result = externalize_ingest_payload(
        content,
        role=role,
        session_id=session_id,
        field_path="content",
        config=config,
        hermes_home=hermes_home,
        kind=_QUARANTINED_ASSISTANT_KIND,
    )
    if result is None:
        logger.warning(
            "LCM ingest protection could not quarantine repetitive assistant output; preserving inline content for lossless recovery"
        )
        return None

    payload = result.get("payload") or {}
    path = result.get("path")
    summary = {
        "ref": getattr(path, "name", ""),
        "kind": payload.get("kind", _QUARANTINED_ASSISTANT_KIND),
        "role": payload.get("role", role),
        "field_path": payload.get("field_path", "content"),
        "content_chars": payload.get("content_chars", len(content)),
        "content_bytes": payload.get("content_bytes", len(content.encode("utf-8"))),
    }
    return _quarantined_assistant_placeholder(summary, reason=reason)


def _existing_quarantined_assistant_placeholder(
    content: str,
    *,
    role: str,
    session_id: str,
    config,
    hermes_home: str,
    reason: str,
) -> str | None:
    existing = find_externalized_payload_for_message(
        content,
        session_id=session_id,
        kind=_QUARANTINED_ASSISTANT_KIND,
        role=role,
        config=config,
        hermes_home=hermes_home,
    )
    if existing is None:
        return None
    return _quarantined_assistant_placeholder(existing, reason=reason)


def restore_ingest_payload_placeholders(
    text: str,
    *,
    config,
    hermes_home: str = "",
    session_id: str = "",
) -> str:
    """Restore ingest placeholders in a stored identity string for matching only.

    Missing or mismatched payload files leave the placeholder untouched so callers
    never fabricate content or hide a recovery problem.
    """
    if not isinstance(text, str) or _EXTERNALIZED_PLACEHOLDER_PREFIX not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        payload = load_externalized_payload(ref, config=config, hermes_home=hermes_home)
        if payload is None or payload.get("kind") != "ingest_payload":
            return match.group(0)
        payload_session_id = payload.get("session_id") or ""
        if session_id and payload_session_id and payload_session_id != session_id:
            return match.group(0)
        content = payload.get("content")
        return content if isinstance(content, str) else match.group(0)

    return _INGEST_PLACEHOLDER_RE.sub(replace, text)


def looks_like_long_base64(text: str, *, min_chars: int = _GENERIC_BASE64_MIN_CHARS) -> bool:
    """Conservative long-base64 heuristic.

    Avoids short hashes/IDs/JWT-ish snippets by requiring a very long run and a
    high base64 alphabet ratio. PEM blocks and ordinary logs contain delimiters
    or whitespace/headers that keep them from matching as one clean run.
    """
    if not isinstance(text, str) or len(text) < min_chars:
        return False
    compact = "".join(text.split())
    if len(compact) < min_chars:
        return False
    if len(compact) % 4 == 1:
        return False
    if not _BASE64_ALPHABET_RE.match(text):
        return False
    # Compute the base64 density over the whitespace-stripped content, not the
    # raw text: otherwise line-ending overhead sinks the ratio and canonical
    # CRLF-wrapped MIME (76/78 = 0.974) and PEM (64/66 = 0.970) blocks fall
    # below 0.98 and are wrongly left inline.
    base64_chars = sum(1 for ch in compact if ch.isalnum() or ch in "+/=_-")
    ratio = base64_chars / max(1, len(compact))
    if ratio < 0.98:
        return False
    # Require at least a bit of mixed alphabet so a long log line of one
    # repeated character is not treated as a binary payload.
    return len(set(compact.rstrip("="))) >= 8


def _placeholder_for_payload(
    payload: str,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
) -> str | None:
    result = externalize_ingest_payload(
        payload,
        role=role,
        session_id=session_id,
        field_path=field_path,
        config=config,
        hermes_home=hermes_home,
    )
    if result is None:
        logger.warning(
            "LCM ingest protection could not externalize payload at %s; preserving inline content for lossless recovery",
            field_path,
        )
        return None
    return result["placeholder"]


def _protect_payload_substrings(
    text: str,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
) -> str:
    if not text or is_externalized_ingest_placeholder(text):
        return text

    def replace_data_uri(match: re.Match[str]) -> str:
        payload = match.group(0)
        return _placeholder_for_payload(
            payload,
            role=role,
            session_id=session_id,
            field_path=field_path,
            config=config,
            hermes_home=hermes_home,
        ) or payload

    protected = _DATA_URI_BASE64_RE.sub(replace_data_uri, text)

    def replace_base64_run(match: re.Match[str]) -> str:
        payload = match.group(1)
        if not looks_like_long_base64(payload):
            return payload
        return _placeholder_for_payload(
            payload,
            role=role,
            session_id=session_id,
            field_path=field_path,
            config=config,
            hermes_home=hermes_home,
        ) or payload

    protected = _BASE64_RUN_RE.sub(replace_base64_run, protected)

    def replace_wrapped_base64(payload: str) -> str:
        return _placeholder_for_payload(
            payload,
            role=role,
            session_id=session_id,
            field_path=field_path,
            config=config,
            hermes_home=hermes_home,
        ) or payload

    # Line-wrapped base64 (MIME/PEM) is not a single contiguous run; externalize
    # it here too so it does not land inline in SQLite/FTS/WAL/backups.
    return _replace_wrapped_base64_blocks(protected, replace_wrapped_base64)


def _maybe_parse_json_string(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    candidates = [text]
    if '\\"' in stripped:
        candidates.append(stripped.replace('\\"', '"'))
    for candidate in candidates:
        if _json_has_duplicate_object_keys(candidate):
            return None
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _json_has_duplicate_object_keys(text: str) -> bool:
    stripped = text.strip() if isinstance(text, str) else ""
    if not stripped or stripped[0] not in "[{":
        return False
    duplicate = False

    def detect_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        seen: set[str] = set()
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                duplicate = True
            seen.add(key)
            result[key] = value
        return result

    try:
        json.loads(text, object_pairs_hook=detect_pairs)
    except Exception:
        return False
    return duplicate


def _dict_field_path(parent: str, key: Any) -> str:
    component = str(key)
    return f"{parent}.{component}" if parent else component


def _payload_key_field_path(parent: str) -> str:
    return f"{parent}.<key>" if parent else "<key>"


def _protect_value(
    value: Any,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
    parse_json_strings: bool = False,
) -> Any:
    value = redact_sensitive_value(
        value,
        config,
        parse_json_strings=parse_json_strings,
    )
    if isinstance(value, dict):
        protected: dict[Any, Any] = {}
        for key, val in value.items():
            protected_key = (
                _protect_payload_substrings(
                    key,
                    role=role,
                    session_id=session_id,
                    field_path=_payload_key_field_path(field_path),
                    config=config,
                    hermes_home=hermes_home,
                )
                if isinstance(key, str)
                else key
            )
            child_path_key = "<key>" if protected_key != key else protected_key
            protected[protected_key] = _protect_value(
                val,
                role=role,
                session_id=session_id,
                field_path=_dict_field_path(field_path, child_path_key),
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=parse_json_strings,
            )
        return protected
    if isinstance(value, list):
        return [
            _protect_value(
                item,
                role=role,
                session_id=session_id,
                field_path=f"{field_path}[{idx}]",
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=parse_json_strings,
            )
            for idx, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    if parse_json_strings:
        if _json_has_duplicate_object_keys(value):
            return _protect_payload_substrings(
                value,
                role=role,
                session_id=session_id,
                field_path=field_path,
                config=config,
                hermes_home=hermes_home,
            )
        parsed = _maybe_parse_json_string(value)
        if parsed is not None:
            canonical = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            if canonical != value:
                raw_protected = _protect_payload_substrings(
                    value,
                    role=role,
                    session_id=session_id,
                    field_path=field_path,
                    config=config,
                    hermes_home=hermes_home,
                )
                if raw_protected != value:
                    return raw_protected
            protected = _protect_value(
                parsed,
                role=role,
                session_id=session_id,
                field_path=field_path,
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=True,
            )
            if protected != parsed:
                return json.dumps(protected, ensure_ascii=False, separators=(",", ":"))
            return value

    return _protect_payload_substrings(
        value,
        role=role,
        session_id=session_id,
        field_path=field_path,
        config=config,
        hermes_home=hermes_home,
    )


def protect_inline_payloads_in_text(
    text: str,
    *,
    role: str,
    session_id: str,
    field_path: str,
    config,
    hermes_home: str,
) -> str:
    """Externalize inline media/base64 payloads inside a text scaffold.

    This is used for non-SQLite active-context scaffolds that still must not
    duplicate media-ish payloads into summaries or preserved objective text.
    """
    if not isinstance(text, str):
        return text
    text = redact_sensitive_text(text, config)
    return _protect_payload_substrings(
        text,
        role=role,
        session_id=session_id,
        field_path=field_path,
        config=config,
        hermes_home=hermes_home,
    )


def _protect_tool_calls(tool_calls: Any, *, role: str, session_id: str, config, hermes_home: str) -> Any:
    return _protect_value(
        tool_calls,
        role=role,
        session_id=session_id,
        field_path="tool_calls",
        config=config,
        hermes_home=hermes_home,
        parse_json_strings=True,
    )


def protect_message_for_ingest(
    message: Dict[str, Any],
    config,
    hermes_home: str = "",
    session_id: str = "",
) -> Dict[str, Any]:
    """Return a copy of ``message`` safe to persist in SQLite.

    Payloads are externalized losslessly when they are inline media/base64-like
    strings before they hit ``messages.content`` or ``messages.tool_calls``.
    When the opt-in generic large-output externalization setting is enabled,
    whole-message content still follows the existing threshold-based behavior.
    """
    msg = dict(message or {})
    role = str(msg.get("role") or "unknown")
    raw_content = msg.get("content")
    raw_normalized_content = normalize_content_value(raw_content)
    original_content = redact_sensitive_value(
        raw_content,
        config,
        parse_json_strings=False,
    )
    normalized_content = normalize_content_value(original_content)
    recovered_with_stat = recover_hermes_persisted_output_with_file_stat(raw_normalized_content) if role == "tool" else None
    recovered_file_stat = None
    recovered_externalized = None
    if recovered_with_stat is not None:
        recovered_persisted_output, recovered_file_stat = recovered_with_stat
        recovered_content = redact_sensitive_value(
            recovered_persisted_output,
            config,
            parse_json_strings=False,
        )
        normalized_recovered_content = normalize_content_value(recovered_content)
        if normalized_recovered_content:
            persisted_output_source_path = _persisted_output_saved_path(raw_normalized_content)
            persisted_output_preview_sha256 = _persisted_output_preview_prefix_digest(raw_normalized_content)
            if _has_lossy_sensitive_redaction(normalized_content):
                persisted_output_preview_sha256 = None
            persisted_output_metadata = {
                "persisted_output_source_path": persisted_output_source_path,
                "persisted_output_expected_chars": _expected_persisted_output_chars(raw_normalized_content),
                "persisted_output_redacted_preview_sha256": _persisted_output_preview_prefix_digest(normalized_content),
                "persisted_output_file_size": recovered_file_stat["size"],
                "persisted_output_file_mtime_ns": recovered_file_stat["mtime_ns"],
                "persisted_output_file_ctime_ns": recovered_file_stat["ctime_ns"],
            }
            if persisted_output_preview_sha256:
                persisted_output_metadata["persisted_output_preview_sha256"] = persisted_output_preview_sha256
            recovered_externalized = maybe_externalize_payload(
                normalized_recovered_content,
                kind="tool_result",
                tool_call_id=str(msg.get("tool_call_id") or ""),
                session_id=session_id,
                role=role,
                config=config,
                hermes_home=hermes_home,
                force=True,
                metadata=persisted_output_metadata,
            )

    # A host-side truncation marker without durable recovered storage is not
    # lossless. Keep the marker/preview visible inline instead of hiding it
    # behind an LCM externalized-payload ref that would look recoverable.
    preserve_truncation_marker_inline = (
        role == "tool"
        and recovered_externalized is None
        and isinstance(normalized_content, str)
        and (
            _is_hermes_persisted_output_marker(normalized_content)
            or _is_unrecoverable_tool_truncation_marker(normalized_content)
        )
    )

    # Preserve the pre-existing opt-in large-output behavior on message content.
    # The always-on storage-boundary sanitizer below is a narrower safety net for
    # inline media/base64 substrings, including cases below the generic threshold
    # or when generic externalization is disabled.
    if normalized_content:
        if recovered_externalized:
            msg["content"] = recovered_externalized["placeholder"]
        elif (
            is_externalized_ingest_placeholder(normalized_content)
            or is_externalized_placeholder(normalized_content)
        ):
            msg["content"] = original_content
        elif preserve_truncation_marker_inline:
            protected_content = _protect_value(
                original_content,
                role=role,
                session_id=session_id,
                field_path="content",
                config=config,
                hermes_home=hermes_home,
                parse_json_strings=False,
            )
            if (
                role == "tool"
                and not bool(getattr(config, "large_output_externalization_enabled", True))
                and _is_hermes_persisted_output_marker(raw_normalized_content)
            ):
                protected_content = _add_inline_persisted_output_identity_metadata(
                    normalize_content_value(protected_content) or "",
                    _persisted_output_marker_identity_digest(raw_normalized_content),
                )
            if recovered_with_stat is not None and _is_hermes_persisted_output_marker(normalized_content):
                protected_content = _add_inline_persisted_output_generation_metadata(
                    normalize_content_value(protected_content) or "",
                    recovered_file_stat,
                )
            msg["content"] = protected_content
        else:
            reason = (
                assistant_output_quarantine_reason(normalized_content)
                if role == "assistant"
                else None
            )
            externalized = None
            if reason:
                placeholder = _externalize_quarantined_assistant_output(
                    normalized_content,
                    role=role,
                    session_id=session_id,
                    config=config,
                    hermes_home=hermes_home,
                    reason=reason,
                )
                if placeholder:
                    externalized = {"placeholder": placeholder}
            if externalized is None:
                kind = _externalization_kind_for_message(msg)
                externalized = maybe_externalize_payload(
                    normalized_content,
                    kind=kind,
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    session_id=session_id,
                    role=role,
                    config=config,
                    hermes_home=hermes_home,
                )
            if externalized:
                msg["content"] = externalized["placeholder"]
            else:
                msg["content"] = _protect_value(
                    original_content,
                    role=role,
                    session_id=session_id,
                    field_path="content",
                    config=config,
                    hermes_home=hermes_home,
                    parse_json_strings=False,
                )
    else:
        msg["content"] = original_content

    if msg.get("tool_calls"):
        msg["tool_calls"] = _protect_tool_calls(
            msg.get("tool_calls"),
            role=role,
            session_id=session_id,
            config=config,
            hermes_home=hermes_home,
        )

    return msg


def quarantine_suspicious_assistant_message(
    message: Dict[str, Any],
    config,
    hermes_home: str = "",
    session_id: str = "",
    *,
    externalize: bool = True,
    prefer_existing_externalized: bool = False,
) -> Dict[str, Any]:
    """Return ``message`` with obviously broken assistant output quarantined.

    Unlike full ingest protection, this only touches suspicious assistant text.
    It is safe for active-context replay because it does not externalize user
    media, tool results, or ordinary long content.
    """
    msg = dict(message or {})
    role = str(msg.get("role") or "unknown")
    if role != "assistant":
        return msg
    normalized_content = normalize_content_value(msg.get("content"))
    reason = assistant_output_quarantine_reason(normalized_content)
    if not reason:
        return msg
    if externalize:
        placeholder = _externalize_quarantined_assistant_output(
            normalized_content,
            role=role,
            session_id=session_id,
            config=config,
            hermes_home=hermes_home,
            reason=reason,
        )
    else:
        placeholder = None
        if prefer_existing_externalized:
            placeholder = _existing_quarantined_assistant_placeholder(
                normalized_content,
                role=role,
                session_id=session_id,
                config=config,
                hermes_home=hermes_home,
                reason=reason,
            )
        if placeholder is None:
            placeholder = _volatile_quarantined_assistant_placeholder(
                normalized_content,
                reason=reason,
            )
    if not placeholder:
        return msg
    msg["content"] = placeholder
    return msg


def quarantine_suspicious_assistant_messages(
    messages: List[Dict[str, Any]],
    config,
    hermes_home: str = "",
    session_id: str = "",
    externalize: List[bool] | None = None,
    prefer_existing_externalized: List[bool] | None = None,
) -> List[Dict[str, Any]]:
    return [
        quarantine_suspicious_assistant_message(
            message,
            config=config,
            hermes_home=hermes_home,
            session_id=session_id,
            externalize=True if externalize is None else externalize[idx],
            prefer_existing_externalized=False
            if prefer_existing_externalized is None
            else prefer_existing_externalized[idx],
        )
        for idx, message in enumerate(messages)
    ]


def protect_messages_for_ingest(
    messages: List[Dict[str, Any]],
    config,
    hermes_home: str = "",
    session_id: str = "",
) -> List[Dict[str, Any]]:
    return [
        protect_message_for_ingest(
            message,
            config=config,
            hermes_home=hermes_home,
            session_id=session_id,
        )
        for message in messages
    ]


def _append_unique_refs(target: list[str], refs: list[str]) -> None:
    for ref in refs:
        if ref not in target:
            target.append(ref)


def _walk_string_values(value: Any):
    if isinstance(value, str):
        yield value
        parsed = _maybe_parse_json_string(value)
        if parsed is not None and not (isinstance(parsed, str) and parsed == value):
            yield from _walk_string_values(parsed)
    elif isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                yield key
            yield from _walk_string_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_string_values(item)


def _walk_tool_call_argument_values(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "arguments":
                yield nested
            yield from _walk_tool_call_argument_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_tool_call_argument_values(item)


def _is_inside_token_quote_span(text: str, start: int, token: str) -> bool:
    in_span = False
    i = 0
    while i < start:
        if text.startswith(token, i):
            in_span = not in_span
            i += len(token)
        else:
            i += 1
    return in_span


def _looks_like_example_quote_context(context: str) -> bool:
    return re.search(r"(?:pytest\s+output|log|example|traceback|failure)\s*:\s*$", context.lower()) is not None


def _has_local_escaped_quote_before(text: str, start: int) -> bool:
    boundary = max(text.rfind(delimiter, 0, start) for delimiter in (",", "{", "["))
    segment = text[boundary + 1:start]
    matches = list(re.finditer(r"\\+[\"']", segment))
    if not matches:
        return False
    quote = matches[-1]
    context = segment[max(0, quote.start() - 80):quote.start()]
    return _looks_like_example_quote_context(context)


def _is_escaped_placeholder_example(text: str, start: int) -> bool:
    prefix = text[max(0, start - 8):start]
    return prefix.endswith("\\") or _has_local_escaped_quote_before(text, start)


def _is_quoted_placeholder_example(text: str, start: int) -> bool:
    for quote_token in ('"', "'"):
        if not _is_inside_token_quote_span(text, start, quote_token):
            continue
        quote = text.rfind(quote_token, 0, start)
        if quote < 0:
            continue
        context = text[max(0, quote - 80):quote]
        if _looks_like_example_quote_context(context):
            return True
    return False


def _looks_like_json_container_string(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _looks_like_example_payload_ref(ref: str) -> bool:
    name = Path(ref).name.lower()
    return name.startswith(("example-", "example_", "fake-", "fake_", "dummy-", "dummy_", "placeholder-", "placeholder_"))


def _extract_unescaped_externalized_payload_refs(text: str, *, ignore_quoted_spans: bool = False) -> list[str]:
    refs: list[str] = []
    for pattern in (_INGEST_PLACEHOLDER_RE, _EXTERNALIZED_PAYLOAD_PLACEHOLDER_RE):
        for match in pattern.finditer(text):
            ref = match.group(1).strip()
            if not _is_basename_ref(ref):
                continue
            if _looks_like_example_payload_ref(ref) and _is_escaped_placeholder_example(text, match.start()):
                continue
            if (
                ignore_quoted_spans
                and _looks_like_example_payload_ref(ref)
                and _is_quoted_placeholder_example(text, match.start())
            ):
                continue
            if ref not in refs:
                refs.append(ref)
    return refs


def _is_quoted_source_literal_match(text: str, match: re.Match[str]) -> bool:
    """Return True when a placeholder match is directly quoted as source text."""
    if match.start() <= 0 or match.end() >= len(text):
        return False
    quote = text[match.start() - 1]
    return quote in ('"', "'") and text[match.end()] == quote


def _extract_source_text_externalized_payload_refs(
    text: str,
    *,
    require_numbered_line: bool = False,
) -> list[str]:
    """Scan source-like text while excluding directly quoted placeholder literals.

    Exact and embedded unquoted placeholders remain owning references.  For
    terminal output, suppression is additionally limited to numbered source
    lines, so ordinary command output remains fail-closed.
    """
    refs: list[str] = []
    for line in text.splitlines() or [text]:
        numbered_source_line = re.match(r"^\s*\d+:\s+", line) is not None
        for pattern in (_INGEST_PLACEHOLDER_RE, _EXTERNALIZED_PAYLOAD_PLACEHOLDER_RE):
            for match in pattern.finditer(line):
                ref = match.group(1).strip()
                if not _is_basename_ref(ref):
                    continue
                if (
                    (not require_numbered_line or numbered_source_line)
                    and _is_quoted_source_literal_match(line, match)
                ):
                    continue
                if ref not in refs:
                    refs.append(ref)
    return refs


def _parse_diagnostic_json(text: str) -> Any:
    """Reject ambiguous envelopes rather than discard duplicate source fields."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate diagnostic key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("non-JSON diagnostic constant")

    if not isinstance(text, str) or len(text) > 2_000_000:
        return None
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError):
        return None


def _diagnostic_source_line_refs(text: str, source_lines: set[int]) -> list[str]:
    """Mask string literals only on independently identified source lines.

    Decode tool envelopes before calling this: scanning their JSON serialization
    can join split fixture strings across escaped newlines. This is diagnostic
    provenance, NOT the extraction/ownership policy used by ingest or readback.
    """
    quoted = re.compile(r'''(?<![\w\\])(?:[rRuUbBfF]{0,2})(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')''')
    refs: list[str] = []
    for index, line in enumerate(text.splitlines()):
        if index in source_lines:
            line = quoted.sub(lambda match: " " * len(match.group()), line)
        _append_unique_refs(refs, extract_all_externalized_payload_refs(line))
    return refs


def _diagnostic_diff_source_lines(lines: list[str]) -> set[int]:
    """Recognize complete unified-diff hunks, not arbitrary +/- output."""
    source_lines: set[int] = set()
    has_headers = False
    for index, line in enumerate(lines):
        if line.startswith("--- "):
            has_headers = index + 1 < len(lines) and lines[index + 1].startswith("+++ ")
        hunk = re.match(r"^@@ -\d{1,9}(?:,(\d{1,9}))? \+\d{1,9}(?:,(\d{1,9}))? @@(?:.*)$", line)
        if not has_headers or hunk is None:
            continue
        old, new = (int(count) if count is not None else 1 for count in hunk.groups())
        candidates: set[int] = set()
        cursor = index + 1
        while (old or new) and cursor < len(lines):
            body = lines[cursor]
            if body == "\\ No newline at end of file":
                cursor += 1
                continue
            if not body or body[0] not in " +-":
                break
            old -= body[0] in " -"
            new -= body[0] in " +"
            if old < 0 or new < 0:
                break
            candidates.add(cursor)
            cursor += 1
        if old == new == 0:
            source_lines.update(candidates)
    return source_lines


def _diagnostic_rtk_diff_source_lines(lines: list[str]) -> set[int]:
    """Accept RTK compact hunks only behind its stats/Changes/footer wrapper.

    Unlike a raw incomplete diff, an explicit additions-omitted sentinel can
    close a hunk, but only when it accounts for exactly its missing additions.
    Paths come from the diffstat, never from a filename/extension allowlist.
    """
    if len(lines) > 20_000:
        return set()
    footer = "[full diff: rtk git diff --no-compact]"
    try:
        end = lines.index(footer)
        changes = lines.index("Changes:", 0, end)
    except ValueError:
        return set()
    preamble = [line for line in lines[:changes] if line.strip()]
    if len(preamble) < 2:
        return set()
    summary = re.fullmatch(
        r"\s*(\d{1,9}) files? changed(?:, \d{1,9} insertions?\(\+\))?(?:, \d{1,9} deletions?\(-\))?", preamble[-1],
    )
    stats = [re.fullmatch(r"\s*(\S.*?)\s+\|\s+\d{1,9}(?:\s+[+-]+)?\s*", line) for line in preamble[:-1]]
    if summary is None or not all(stats) or int(summary[1]) != len(stats):
        return set()
    paths = {match[1].rstrip() for match in stats if match is not None}
    if len(paths) != len(stats):
        return set()
    source_lines: set[int] = set()
    active_path = False
    cursor = changes + 1
    while cursor < end:
        line = lines[cursor]
        if line in paths:
            active_path = True
            cursor += 1
            continue
        hunk = re.fullmatch(r"@@ -\d{1,9},(\d{1,9}) \+\d{1,9},(\d{1,9}) @@(?:.*)", line)
        if not active_path or hunk is None:
            # Blank separators and per-file counts do not grant source status.
            if line.strip() and re.fullmatch(r"  \+\d{1,9} -\d{1,9}", line) is None:
                active_path = False
            cursor += 1
            continue
        old, new = map(int, hunk.groups())
        candidates: set[int] = set()
        cursor += 1
        while (old or new) and cursor < end:
            body = lines[cursor]
            omission = re.fullmatch(r"  \.\.\. \((\d{1,9}) additions? truncated\)", body)
            if omission is not None:
                # RTK also omits trailing unchanged context. The remaining
                # old/new counts must differ by exactly the stated additions.
                if old >= 0 and new - old == int(omission[1]) and new > old:
                    old = new = 0
                    cursor += 1
                break
            if body == "\\ No newline at end of file":
                cursor += 1
                continue
            if not body or body[0] not in " +-" or re.fullmatch(r"  \+\d{1,9} -\d{1,9}", body):
                break
            old -= body[0] in " -"
            new -= body[0] in " +"
            if old < 0 or new < 0:
                break
            candidates.add(cursor)
            cursor += 1
        if old == new == 0:
            source_lines.update(candidates)
        else:
            active_path = False
    return source_lines


def _diagnostic_plain_refs(value: Any) -> list[str]:
    """Scan non-source fields as data, including nested diagnostic messages."""
    refs: list[str] = []
    for text in _walk_string_values(value):
        _append_unique_refs(refs, extract_all_externalized_payload_refs(text))
    return refs


def _refs_from_patch_result(value: Any) -> list[str] | None:
    required = {"success", "diff", "files_modified"}
    allowed = required | {"resolved_path", "lint", "_warning", "error"}
    if not isinstance(value, dict) or not required.issubset(value) or not set(value).issubset(allowed):
        return None
    if value["success"] is not True or not isinstance(value["diff"], str):
        return None
    if not isinstance(value["files_modified"], list) or not all(isinstance(path, str) for path in value["files_modified"]):
        return None
    if any(key in value and not isinstance(value[key], str) for key in ("resolved_path", "_warning")):
        return None
    if value.get("error") is not None and not isinstance(value["error"], str):
        return None
    lint = value.get("lint")
    if lint is not None and (
        not isinstance(lint, dict) or set(lint) != {"status", "output"}
        or not all(isinstance(item, str) for item in lint.values())
    ):
        return None
    refs = _diagnostic_source_line_refs(
        value["diff"], _diagnostic_diff_source_lines(value["diff"].splitlines()),
    )
    _append_unique_refs(refs, _diagnostic_plain_refs({key: item for key, item in value.items() if key != "diff"}))
    return refs


def _refs_from_read_file_result(value: Any) -> list[str] | None:
    required = {"content", "total_lines", "file_size", "truncated", "is_binary", "is_image"}
    allowed = required | {"hint", "next_offset", "error"}
    if not isinstance(value, dict) or not required.issubset(value) or not set(value).issubset(allowed):
        return None
    if not isinstance(value["content"], str) or type(value["truncated"]) is not bool:
        return None
    if value["is_binary"] is not False or value["is_image"] is not False:
        return None
    if any(type(value[key]) is not int or value[key] < 0 for key in ("total_lines", "file_size")):
        return None
    if "next_offset" in value and (type(value["next_offset"]) is not int or value["next_offset"] < 1):
        return None
    if any(key in value and value[key] is not None and not isinstance(value[key], str) for key in ("hint", "error")):
        return None
    source_lines = {index for index, line in enumerate(value["content"].splitlines()) if re.match(r"^\d+\|", line)}
    refs = _diagnostic_source_line_refs(value["content"], source_lines)
    _append_unique_refs(refs, _diagnostic_plain_refs({key: item for key, item in value.items() if key != "content"}))
    return refs


def _refs_from_diagnostic_excerpt_row(value: Any, *, depth: int) -> list[str] | None:
    """Metadata is not source provenance for truncated head/tail excerpts.

    Only an excerpt covering the entire original field can delegate to the
    ordinary exact-envelope parser. Never join head and tail or guess missing
    JSON, diff headers, or duplicate/unknown fields from a partial patch reply.
    """
    fields = {"store_id", "role", "source", "tool_name", "content_len", "content_head", "content_tail"}
    if not isinstance(value, dict) or set(value) != fields:
        return None
    if (
        type(value["store_id"]) is not int or type(value["content_len"]) is not int
        or value["content_len"] < 0
        or not all(isinstance(value[key], str) for key in fields - {"store_id", "content_len"})
        or any(len(value[key]) > value["content_len"] for key in ("content_head", "content_tail"))
    ):
        return None
    refs = _diagnostic_plain_refs({key: item for key, item in value.items() if key not in {"content_head", "content_tail"}})
    for key in ("content_head", "content_tail"):
        excerpt = value[key]
        if len(excerpt) == value["content_len"]:
            nested = _refs_for_externalized_integrity_scan(
                excerpt, role=value["role"], field="content",
                tool_name=value["tool_name"], _depth=depth + 1,
            )
        else:
            nested = _diagnostic_plain_refs(excerpt)
        _append_unique_refs(refs, nested)
    return refs


def _parse_diagnostic_repr(text: str) -> str | None:
    """Read one bounded, canonical Python repr string; never evaluate code."""
    if not text or len(text) > 2_000_000 or text[0] not in "\"'":
        return None
    import ast

    try:
        node = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and repr(node.value) == text:
        return node.value
    return None


def _refs_from_diagnostic_source_slice(text: str) -> list[str]:
    """Scan repr-decoded slices without pretending a partial JSON is complete.

    Only direct numbered lines or *complete* JSON-escaped numbered lines have
    local source provenance. Decode at most one JSON layer. All fragments and
    unnumbered text, including ordinary quoted owners, keep the plain scan.
    """
    if is_externalized_ingest_placeholder(text.strip()) or is_externalized_placeholder(text.strip()):
        return extract_all_externalized_payload_refs(text)
    refs: list[str] = []
    pieces: list[str] = []
    cursor = 0
    source_spans: list[tuple[int, int]] = []
    # An odd preceding slash can mean an escaped backslash, not a JSON newline;
    # do not attempt to recover that ambiguous boundary.
    boundary = re.compile(r"(?<!\\)\\n")
    breaks = list(boundary.finditer(text))
    for left, right in zip(breaks, breaks[1:]):
        raw = text[left.end():right.start()]
        if re.match(r"^[ \t]*\d{1,9}:[ \t]+", raw) is None:
            continue
        decoded = _parse_diagnostic_json('"' + raw + '"')
        if not isinstance(decoded, str) or "\n" in decoded or "\r" in decoded:
            continue
        pieces.extend((text[cursor:left.end()], " " * len(raw)))
        cursor = right.start()
        source_spans.append((left.end(), right.start()))
        _append_unique_refs(refs, _diagnostic_source_line_refs(decoded, {0}))
    pieces.append(text[cursor:])
    remainder = "".join(pieces)
    numbered = {index for index, line in enumerate(remainder.splitlines()) if re.match(r"^[ \t]*\d{1,9}:[ \t]+", line)}
    _append_unique_refs(refs, _diagnostic_source_line_refs(remainder, numbered))
    # Keep matches crossing between source and unknown text. A match entirely
    # spanning numbered lines is scanned *after* decoding above, not across
    # escaped newlines that could accidentally join split fixture literals.
    from bisect import bisect_right

    starts = [start for start, _ in source_spans]
    for pattern in (_INGEST_PLACEHOLDER_RE, _EXTERNALIZED_PAYLOAD_PLACEHOLDER_RE):
        for match in pattern.finditer(text):
            left = bisect_right(starts, match.start()) - 1
            right = bisect_right(starts, match.end() - 1) - 1
            starts_in_source = left >= 0 and match.start() < source_spans[left][1]
            ends_in_source = right >= 0 and match.end() <= source_spans[right][1]
            overlaps = starts_in_source or right > left
            if overlaps and not (starts_in_source and ends_in_source):
                _append_unique_refs(refs, extract_all_externalized_payload_refs(match.group()))
    return refs


def _refs_from_diagnostic_patch_arguments(text: str) -> list[str] | None:
    """Recognize a full replace-arguments object, not a fragment or call list."""
    value = _parse_diagnostic_json(text)
    if (
        not isinstance(value, dict) or set(value) != {"mode", "path", "old_string", "new_string"}
        or value["mode"] != "replace" or not all(isinstance(item, str) for item in value.values())
    ):
        return None
    refs = _diagnostic_plain_refs(value["path"])
    for key in ("old_string", "new_string"):
        _append_unique_refs(refs, _diagnostic_source_line_refs(value[key], set(range(len(value[key].splitlines())))))
    return refs


def _refs_from_diagnostic_repr_rows(text: str, *, depth: int) -> dict[int, list[str]]:
    """Authorize repr decoding only beneath exact ROW/PATH diagnostic headers."""
    if depth >= 8 or len(text) > 2_000_000:
        return {}
    recognized: dict[int, list[str]] = {}
    mode = ""
    field_len = 0
    for index, line in enumerate(text.splitlines()):
        if re.fullmatch(r"ROW \d{1,18} role (?:tool tool terminal|assistant tool None)", line):
            mode = "slice"
            continue
        row = re.fullmatch(r"ROW \d{1,18} (content|tool_calls)", line)
        if row:
            mode = row[1]
            continue
        path = re.fullmatch(r"PATH (\$\.output|\$\[0\]\.function\.arguments) len (\d{1,9})", line)
        if path and ((mode == "content" and path[1] == "$.output") or (mode == "tool_calls" and path[1] == "$[0].function.arguments")):
            mode = "output" if mode == "content" else "arguments"
            field_len = int(path[2])
            continue
        decoded = _parse_diagnostic_repr(line) if mode in {"slice", "output", "arguments"} else None
        if decoded is None:
            if line.strip():
                mode = ""
            continue
        if mode == "arguments":
            refs = _refs_from_diagnostic_patch_arguments(decoded) if len(decoded) == field_len else None
            # This PATH names one complete arguments field, not a sequence.
            mode = ""
            if refs is not None:
                recognized[index] = refs
        elif mode == "slice" or len(decoded) <= field_len:
            # When the slice is a complete object, its entire schema (including
            # duplicate and unknown keys) is available: do not bypass it with
            # the partial numbered-line recognizer. A malformed closed object
            # is likewise not partial-source evidence.
            stripped = decoded.strip()
            if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
                refs = _refs_from_diagnostic_echo(_parse_diagnostic_json(decoded), depth=depth + 1)
                recognized[index] = refs if refs is not None else _diagnostic_plain_refs(decoded)
            else:
                recognized[index] = _refs_from_diagnostic_source_slice(decoded)
    return recognized


def _refs_from_diagnostic_echo(value: Any, *, depth: int) -> list[str] | None:
    """Recurse only through complete, recognized source/diagnostic envelopes.

    Unknown keys, truncated JSON, and excessive nesting retain the generic
    fail-closed scan. In particular, a quoted string alone is not provenance.
    """
    if depth >= 8:
        return None
    if isinstance(value, str):
        parsed = _parse_diagnostic_json(value)
        return _refs_from_diagnostic_echo(parsed, depth=depth + 1) if parsed is not None else None
    if isinstance(value, list):
        if len(value) > 4096:
            return None
        combined: list[str] = []
        for item in value:
            nested = _refs_from_diagnostic_excerpt_row(item, depth=depth + 1)
            if nested is None:
                nested = _refs_from_diagnostic_echo(item, depth=depth + 1)
            if nested is None:
                return None
            _append_unique_refs(combined, nested)
        return combined
    if _is_lcm_expand_result(value):
        return _refs_for_externalized_integrity_scan(
            json.dumps(value), role="tool", field="content",
            tool_name="lcm_expand", _depth=depth + 1,
        )
    for parser in (_refs_from_patch_result, _refs_from_read_file_result):
        refs = parser(value)
        if refs is not None:
            return refs
    refs = _refs_from_terminal_source_listing(value, depth=depth + 1)
    if refs is not None:
        return refs
    if _is_code_search_result(value):
        return []
    # Bounded metadata-only diagnostic rows (not arbitrary content wrappers).
    if isinstance(value, dict) and {"store_id", "role", "content"}.issubset(value):
        allowed = {"store_id", "session_id", "source", "role", "tool_name", "content", "tool_calls"}
        if (
            set(value).issubset(allowed)
            and type(value["store_id"]) is int
            and isinstance(value["role"], str)
            and isinstance(value["content"], str)
            and all(item is None or isinstance(item, str) for key, item in value.items() if key != "store_id")
        ):
            refs = None
            if value["role"] == "tool" and not value.get("tool_name"):
                refs = _refs_from_diagnostic_echo(_parse_diagnostic_json(value["content"]), depth=depth + 1)
            if refs is None:
                refs = _refs_for_externalized_integrity_scan(
                    value["content"], role=value["role"], field="content",
                    tool_name=value.get("tool_name") or "", _depth=depth + 1,
                )
            _append_unique_refs(refs, _refs_for_externalized_integrity_scan(
                value.get("tool_calls") or "", role=value["role"], field="tool_calls", _depth=depth + 1,
            ))
            _append_unique_refs(refs, _diagnostic_plain_refs({
                key: item for key, item in value.items() if key not in {"content", "tool_calls"}
            }))
            return refs
    return None


def _refs_from_diagnostic_terminal_output(text: str, *, depth: int) -> list[str]:
    nested = _refs_from_diagnostic_echo(_parse_diagnostic_json(text), depth=depth)
    if nested is not None:
        return nested
    lines = text.splitlines()
    source_lines = _diagnostic_diff_source_lines(lines)
    source_lines.update(_diagnostic_rtk_diff_source_lines(lines))
    repr_rows = _refs_from_diagnostic_repr_rows(text, depth=depth)
    in_failures = False
    in_test_source = False
    for index, line in enumerate(lines):
        if re.match(r"^=+ FAILURES =+$", line):
            in_failures = True
        elif re.match(r"^=+ .* =+$", line):
            in_failures = in_test_source = False
        if in_failures and re.match(r"^    (?:async )?def test_\w+\(", line):
            in_test_source = True
        if in_test_source:
            if line.startswith(("    ", ">   ")):
                source_lines.add(index)
            elif line.strip():
                in_test_source = False
        if re.match(r"^\s*\d+:\s+", line):
            source_lines.add(index)
    refs: list[str] = []
    for index, line in enumerate(lines):
        if index in repr_rows:
            _append_unique_refs(refs, repr_rows[index])
            continue
        if index not in source_lines:
            nested = _refs_from_diagnostic_echo(_parse_diagnostic_json(line), depth=depth)
            if nested is not None:
                _append_unique_refs(refs, nested)
                continue
        _append_unique_refs(refs, _diagnostic_source_line_refs(line, {0} if index in source_lines else set()))
    return refs


def _refs_from_terminal_source_listing(value: Any, *, depth: int = 0) -> list[str] | None:
    """Return refs for an exact terminal result shape, or None when uncertain."""
    if not isinstance(value, dict):
        return None
    required = {"output", "exit_code", "error"}
    allowed = required | {"approval"}
    if not required.issubset(value) or not set(value).issubset(allowed):
        return None
    if not isinstance(value["output"], str):
        return None
    if not isinstance(value["exit_code"], int) or isinstance(value["exit_code"], bool):
        return None
    if value["error"] is not None and not isinstance(value["error"], str):
        return None
    if "approval" in value and not isinstance(value["approval"], str):
        return None

    refs = _refs_from_diagnostic_terminal_output(value["output"], depth=depth)
    for key in ("error", "approval"):
        nested = value.get(key)
        if isinstance(nested, str):
            _append_unique_refs(refs, _extract_unescaped_externalized_payload_refs(nested))
    return refs


def _refs_from_single_patch_tool_call(value: Any) -> list[str] | None:
    """Exclude quoted source literals from one exact ``patch`` replace call.

    Returning None deliberately falls back to the generic fail-closed scanner
    for mixed calls, provider extensions, malformed JSON, or unknown schemas.
    """
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        return None
    call = value[0]
    allowed_call_keys = {"id", "call_id", "response_item_id", "type", "function"}
    if (
        not {"type", "function"}.issubset(call)
        or not set(call).issubset(allowed_call_keys)
        or call.get("type") != "function"
    ):
        return None
    if any(
        key in call and call[key] is not None and not isinstance(call[key], str)
        for key in ("id", "call_id", "response_item_id")
    ):
        return None
    function = call.get("function")
    if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
        return None
    if function.get("name") != "patch" or not isinstance(function.get("arguments"), str):
        return None
    arguments = _maybe_parse_json_string(function["arguments"])
    if not isinstance(arguments, dict) or set(arguments) != {
        "mode",
        "path",
        "old_string",
        "new_string",
    }:
        return None
    if arguments.get("mode") != "replace" or not all(
        isinstance(arguments.get(key), str) for key in ("path", "old_string", "new_string")
    ):
        return None

    refs: list[str] = []
    for key in ("old_string", "new_string"):
        _append_unique_refs(
            refs,
            _extract_source_text_externalized_payload_refs(arguments[key]),
        )
    return refs


def _is_code_search_result(value: Any) -> bool:
    """Return True only for the structured result emitted by ``search_files``.

    Placeholder-shaped text in ``matches_text`` is source-code evidence, not a
    reference from the message that stores the tool result. Keep this structural
    check deliberately exact so ordinary JSON tool results continue to be
    scanned for externalized payload refs.
    """
    if not isinstance(value, dict) or set(value) != {
        "total_count",
        "matches_format",
        "matches_text",
    }:
        return False
    matches_format = value.get("matches_format")
    return (
        isinstance(value.get("total_count"), int)
        and not isinstance(value.get("total_count"), bool)
        and isinstance(matches_format, str)
        and matches_format.startswith("path-grouped:")
        and "<line>: <content>" in matches_format
        and isinstance(value.get("matches_text"), str)
    )


def _is_lcm_expand_result(value: Any) -> bool:
    """Recognize raw-message recall metadata without trusting provider extensions."""
    string_keys = {
        "source_type", "role", "content", "session_id", "source", "conversation_id",
        "tool_call_id", "tool_name", "exact_ref", "externalized_ref", "externalized_note",
    }
    integer_keys = {"store_id", "content_chars", "content_offset", "content_returned_chars", "next_content_offset"}
    boolean_keys = {"from_current_session", "content_truncated", "has_more"}
    allowed = string_keys | integer_keys | boolean_keys | {"timestamp", "externalized_refs", "externalized", "externalized_payloads"}
    return (
        isinstance(value, dict)
        and {"source_type", "store_id", "role", "content"}.issubset(value)
        and set(value).issubset(allowed)
        and value["source_type"] == "raw_message"
        and all(isinstance(value[key], str) for key in string_keys & value.keys())
        and all(type(value[key]) is int for key in integer_keys & value.keys())
        and all(type(value[key]) is bool for key in boolean_keys & value.keys())
        and ("timestamp" not in value or type(value["timestamp"]) in (int, float))
        and isinstance(value.get("externalized_refs", []), list)
        and all(isinstance(ref, str) for ref in value.get("externalized_refs", []))
        and isinstance(value.get("externalized", {}), dict)
        and isinstance(value.get("externalized_payloads", []), list)
    )


def _refs_for_externalized_integrity_scan(
    value: str,
    *,
    role: str,
    field: str,
    tool_name: str = "",
    _depth: int = 0,
) -> list[str]:
    """Return refs that plausibly came from LCM storage-boundary placeholders.

    Tool outputs and tool-call arguments often contain escaped code snippets,
    pytest failures, or docs that mention placeholder examples. Counting those
    as live payload references turns doctor into a false-positive machine. Exact
    placeholders are still counted everywhere; embedded unescaped placeholders
    are counted for message content, raw JSON-container tool-call argument
    strings, and raw free-form tool-call argument strings so ingestion-produced
    refs do not disappear while quoted examples stay ignored.
    """
    if not isinstance(value, str) or not value:
        return []
    stripped = value.strip()
    if is_externalized_ingest_placeholder(stripped) or is_externalized_placeholder(stripped):
        return extract_all_externalized_payload_refs(stripped)
    if field == "tool_calls":
        parsed = _maybe_parse_json_string(value)
        patch_refs = _refs_from_single_patch_tool_call(parsed)
        if patch_refs is not None:
            return patch_refs
        refs = _extract_unescaped_externalized_payload_refs(value, ignore_quoted_spans=True)
        if parsed is None:
            return refs
        for argument in _walk_tool_call_argument_values(parsed):
            if isinstance(argument, str):
                _append_unique_refs(refs, _extract_unescaped_externalized_payload_refs(argument, ignore_quoted_spans=True))
                parsed_argument = _maybe_parse_json_string(argument)
                if parsed_argument is not None:
                    for nested in _walk_string_values(parsed_argument):
                        nested_stripped = nested.strip()
                        if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                            _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
                        else:
                            _append_unique_refs(
                                refs,
                                _extract_unescaped_externalized_payload_refs(nested, ignore_quoted_spans=True),
                            )
            else:
                for nested in _walk_string_values(argument):
                    nested_stripped = nested.strip()
                    if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                        _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
                    else:
                        _append_unique_refs(refs, _extract_unescaped_externalized_payload_refs(nested, ignore_quoted_spans=True))
        for nested in _walk_string_values(parsed):
            nested_stripped = nested.strip()
            if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
            else:
                _append_unique_refs(refs, _extract_unescaped_externalized_payload_refs(nested, ignore_quoted_spans=True))
        return refs
    if role == "tool":
        parsed = _parse_diagnostic_json(value)
        if field == "content" and _depth < 8 and tool_name in {"patch", "read_file"}:
            source_parser = _refs_from_patch_result if tool_name == "patch" else _refs_from_read_file_result
            source_refs = source_parser(parsed)
            if source_refs is not None:
                return source_refs
        if field == "content" and tool_name == "terminal" and _depth < 8:
            terminal_refs = _refs_from_terminal_source_listing(parsed, depth=_depth + 1)
            if terminal_refs is not None:
                return terminal_refs
        if (
            field == "content"
            and tool_name == "search_files"
            and _is_code_search_result(parsed)
        ):
            return []
        if (
            field == "content"
            and tool_name == "lcm_expand"
            and _depth < 8
            and _is_lcm_expand_result(parsed)
        ):
            assert isinstance(parsed, dict)
            expanded_content = parsed["content"]
            expanded_role = parsed["role"]
            # lcm_expand does not always carry the original tool name. Infer
            # only exact known envelopes, never source status from quoting alone.
            expanded_refs = None
            if expanded_role == "tool":
                expanded_refs = _refs_from_diagnostic_echo(
                    _parse_diagnostic_json(expanded_content), depth=_depth + 1,
                )
            if expanded_refs is None:
                expanded_refs = _refs_for_externalized_integrity_scan(
                    expanded_content, role=expanded_role, field="content",
                    tool_name=parsed.get("tool_name") or "", _depth=_depth + 1,
                )
            _append_unique_refs(expanded_refs, _diagnostic_plain_refs(
                {key: item for key, item in parsed.items() if key != "content"},
            ))
            return expanded_refs
        refs = _extract_unescaped_externalized_payload_refs(value)
        if parsed is not None:
            for nested in _walk_string_values(parsed):
                nested_stripped = nested.strip()
                if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                    _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
                else:
                    _append_unique_refs(refs, _extract_unescaped_externalized_payload_refs(nested, ignore_quoted_spans=True))
        return refs
    return extract_all_externalized_payload_refs(value)


def _refs_from_verified_diagnostic_echo(value: str, conn, *, store_id: int) -> list[str] | None:
    """Resolve exact copied excerpts against their earlier stored source rows.

    Truncation is not itself source provenance. Only matching bytes and row
    metadata allow replacing an excerpt's guesses with its original field's
    diagnostic refs. Actual owning refs in that original field remain counted.
    Unknown, unavailable, edited or future sources retain ordinary scanning.
    """
    outer = _parse_diagnostic_json(value)
    if not isinstance(outer, dict) or not {"output", "exit_code", "error"}.issubset(outer):
        return None
    if not set(outer).issubset({"output", "exit_code", "error", "approval"}):
        return None
    if not isinstance(outer["output"], str) or type(outer["exit_code"]) is not int:
        return None
    if any(outer.get(key) is not None and not isinstance(outer[key], str) for key in ("error", "approval")):
        return None
    output = outer["output"]
    if len(output) > 2_000_000:
        return None
    cache: dict[int, Any] = {}

    def original(source_id):
        if type(source_id) is not int or not 0 < source_id < store_id:
            return None
        if source_id not in cache:
            if len(cache) >= 256:
                return None
            cache[source_id] = conn.execute(
                "SELECT role, source, tool_name, content, tool_calls FROM messages "
                "WHERE store_id=? AND length(COALESCE(content,'')) <= 2000000 "
                "AND length(COALESCE(tool_calls,'')) <= 2000000", (source_id,),
            ).fetchone()
        return cache[source_id]

    def field_refs(row, field):
        text = row[3 if field == "content" else 4] or ""
        return _refs_for_externalized_integrity_scan(
            text, role=str(row[0] or ""), field=field, tool_name=str(row[2] or ""),
        )

    refs = _diagnostic_plain_refs({key: item for key, item in outer.items() if key != "output"})
    parsed = _parse_diagnostic_json(output)
    if isinstance(parsed, list) and 0 < len(parsed) <= 256:
        fields = {"store_id", "role", "source", "tool_name", "content_len", "content_head", "content_tail"}
        for item in parsed:
            if not isinstance(item, dict) or set(item) != fields:
                return None
            if type(item["content_len"]) is not int or not all(isinstance(item[key], str) for key in ("role", "source", "content_head", "content_tail")):
                return None
            row = original(item["store_id"])
            if row is None or tuple(row[:3]) != (item["role"], item["source"], item["tool_name"]):
                return None
            text = row[3] or ""
            # Some diagnostics use a one-based SQL tail offset. Exact substring
            # membership, not a guessed endpoint repair, establishes the copy.
            if len(text) != item["content_len"] or not text.startswith(item["content_head"]) or item["content_tail"] not in text:
                return None
            _append_unique_refs(refs, field_refs(row, "content"))
        return refs
    row = None
    excerpts = 0
    for line in output.splitlines():
        if not line.strip():
            continue
        header = re.fullmatch(r"ROW (\d{1,18}) role (tool|assistant) tool (terminal|None)", line)
        if header:
            if row is not None and not excerpts:
                return None
            row = original(int(header[1]))
            if row is None or row[0] != header[2] or (row[2] or "None") != header[3]:
                return None
            excerpts = 0
            continue
        excerpt = _parse_diagnostic_repr(line)
        if row is None or not excerpt:
            return None
        matching = [field for field, text in (("content", row[3]), ("tool_calls", row[4])) if isinstance(text, str) and excerpt in text]
        if not matching:
            return None
        for field in matching:
            _append_unique_refs(refs, field_refs(row, field))
        excerpts += 1
    return refs if row is not None and excerpts else None


def scan_externalized_payload_integrity(conn, config, *, hermes_home: str = "", limit: int = 5) -> dict[str, Any]:
    """Compare externalized payload refs stored in messages with JSON files.

    This is intentionally read-only and metadata-only. It does not open payload
    files except through directory metadata, and row samples never include raw
    message content or tool-call arguments.
    """

    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    existing_files: set[str] = set()
    if storage_dir.exists() and storage_dir.is_dir():
        existing_files = {path.name for path in storage_dir.glob("*.json") if path.is_file()}

    referenced_refs: set[str] = set()
    first_location_by_ref: dict[str, dict[str, Any]] = {}
    for store_id, session_id, source, role, tool_name, content, tool_calls in conn.execute(
        """
        SELECT store_id, session_id, source, role, tool_name, content, tool_calls
        FROM messages
        WHERE COALESCE(content, '') LIKE '%ref=%]%'
           OR COALESCE(tool_calls, '') LIKE '%ref=%]%'
        ORDER BY store_id ASC
        """
    ).fetchall():
        for field, value in (("content", content), ("tool_calls", tool_calls)):
            if not isinstance(value, str):
                continue
            refs = None
            if field == "content" and role == "tool" and tool_name == "terminal":
                refs = _refs_from_verified_diagnostic_echo(value, conn, store_id=int(store_id))
            if refs is None:
                refs = _refs_for_externalized_integrity_scan(
                    value, role=str(role or ""), field=field, tool_name=str(tool_name or ""),
                )
            for ref in refs:
                referenced_refs.add(ref)
                first_location_by_ref.setdefault(
                    ref,
                    {
                        "store_id": int(store_id),
                        "session_id": session_id,
                        "source": source,
                        "role": role,
                        "field": field,
                        "externalized_ref": ref,
                    },
                )

    missing_refs = sorted(ref for ref in referenced_refs if ref not in existing_files)
    existing_ref_count = sum(1 for ref in referenced_refs if ref in existing_files)
    unreferenced_files = sorted(ref for ref in existing_files if ref not in referenced_refs)

    return {
        "externalized_payload_refs_total": len(referenced_refs),
        "externalized_payload_refs_existing": existing_ref_count,
        "externalized_payload_refs_missing": len(missing_refs),
        "externalized_payload_files_unreferenced": len(unreferenced_files),
        "missing_externalized_payload_refs": [
            first_location_by_ref[ref] for ref in missing_refs[:limit] if ref in first_location_by_ref
        ],
        "unreferenced_externalized_payload_files": [
            {"externalized_ref": ref} for ref in unreferenced_files[:limit]
        ],
    }


def scan_sqlite_payload_risks(conn, *, limit: int = 5) -> dict[str, Any]:
    """Return bounded diagnostics for suspicious inline payload storage.

    Diagnostics intentionally omit previews/raw payload text. Rows include only
    metadata needed for triage and a recoverability ref when a compact
    externalized placeholder is present.
    """

    def make_row(row, *, field: str, length_key: str, category: str) -> dict[str, Any]:
        store_id, session_id, source, role, length, value = row
        value = value or ""
        result = {
            "store_id": int(store_id),
            "session_id": session_id,
            "source": source,
            "role": role,
            "field": field,
            "length": int(length or 0),
            length_key: int(length or 0),
            "suspicious_category": category,
        }
        refs = extract_ingest_externalized_refs(value) if isinstance(value, str) else []
        ref = refs[0] if refs else (extract_externalized_ref(value) if isinstance(value, str) else None)
        if ref:
            result["externalized_ref"] = ref
        return result

    largest_content = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        ORDER BY content_len DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    largest_tool_calls = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(tool_calls), 0) AS tool_calls_len, tool_calls
        FROM messages
        ORDER BY tool_calls_len DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    candidate_cap = max(limit * 20, limit)
    # Pre-filter broadly in SQL, then apply the same conservative Python regex
    # used by ingest externalization. This avoids false positives from code or
    # doctor text that quotes scaffolds such as "data:%;base64,%" or
    # `DATA_URI = "data:image/png;base64," + DATA_PAYLOAD`.
    data_uri_content_candidates = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        WHERE lower(content) GLOB '*data:*;base64,*'
        ORDER BY content_len DESC
        LIMIT ?
        """,
        (candidate_cap,),
    ).fetchall()
    data_uri_tool_call_candidates = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(tool_calls), 0) AS tool_calls_len, tool_calls
        FROM messages
        WHERE lower(tool_calls) GLOB '*data:*;base64,*'
        ORDER BY tool_calls_len DESC
        LIMIT ?
        """,
        (candidate_cap,),
    ).fetchall()
    data_uri_content = [
        row for row in data_uri_content_candidates if isinstance(row[-1], str) and contains_data_uri_base64(row[-1])
    ][:limit]
    data_uri_tool_calls = [
        row for row in data_uri_tool_call_candidates if isinstance(row[-1], str) and contains_data_uri_base64(row[-1])
    ][:limit]

    generic_rows = []
    for store_id, session_id, source, role, content, tool_calls in conn.execute(
        """
        SELECT store_id, session_id, source, role, content, tool_calls
        FROM messages
        WHERE COALESCE(length(content), 0) >= ? OR COALESCE(length(tool_calls), 0) >= ?
        ORDER BY MAX(COALESCE(length(content), 0), COALESCE(length(tool_calls), 0)) DESC
        LIMIT ?
        """,
        (_GENERIC_BASE64_MIN_CHARS, _GENERIC_BASE64_MIN_CHARS, candidate_cap),
    ).fetchall():
        for field, value in (("content", content), ("tool_calls", tool_calls)):
            if isinstance(value, str) and contains_long_base64_run(value):
                result = {
                    "store_id": int(store_id),
                    "session_id": session_id,
                    "source": source,
                    "role": role,
                    "field": field,
                    "length": len(value),
                    "suspicious_category": "base64_like",
                }
                refs = extract_ingest_externalized_refs(value)
                ref = refs[0] if refs else extract_externalized_ref(value)
                if ref:
                    result["externalized_ref"] = ref
                generic_rows.append(result)
                break
        if len(generic_rows) >= limit:
            break

    quarantined_assistant_rows = [
        make_row(row, field="content", length_key="content_len", category=_QUARANTINED_ASSISTANT_KIND)
        for row in conn.execute(
            """
            SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
            FROM messages
            WHERE role = 'assistant'
              AND content LIKE '%Externalized LCM ingest payload:%quarantined_assistant_output%'
            ORDER BY store_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    ]

    suspicious_repetitive_assistant_rows = []
    for row in conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        WHERE role = 'assistant'
          AND COALESCE(length(content), 0) >= ?
          AND content NOT LIKE '%Externalized LCM ingest payload:%quarantined_assistant_output%'
        ORDER BY content_len DESC
        LIMIT ?
        """,
        (_QUARANTINED_ASSISTANT_MIN_CHARS, candidate_cap),
    ).fetchall():
        value = row[-1]
        if isinstance(value, str):
            reason = assistant_output_quarantine_reason(value)
            if reason:
                suspicious_repetitive_assistant_rows.append(
                    make_row(row, field="content", length_key="content_len", category=reason)
                )
        if len(suspicious_repetitive_assistant_rows) >= limit:
            break

    heartbeat_noise_rows = []
    for row in conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        WHERE role IN ('assistant', 'tool', 'system')
          AND COALESCE(length(content), 0) BETWEEN 1 AND ?
          AND (
            lower(trim(content)) GLOB 'still working*'
            OR lower(trim(content)) GLOB 'working on it*'
            OR lower(trim(content)) GLOB 'processing*'
            OR lower(trim(content)) GLOB 'checking*'
            OR lower(trim(content)) GLOB 'one moment*'
            OR lower(trim(content)) GLOB 'ping*'
            OR lower(trim(content)) GLOB 'heartbeat*'
            OR lower(trim(content)) GLOB 'no update*'
          )
        ORDER BY store_id ASC
        LIMIT ?
        """,
        (_HEARTBEAT_NOISE_MAX_CHARS, candidate_cap),
    ).fetchall():
        _store_id, _session_id, _source, role, _length, value = row
        reason = heartbeat_noise_reason(str(role or ""), value if isinstance(value, str) else "")
        if reason:
            heartbeat_noise_rows.append(
                make_row(row, field="content", length_key="content_len", category=reason)
            )
        if len(heartbeat_noise_rows) >= limit:
            break

    return {
        "largest_content_rows": [
            make_row(row, field="content", length_key="content_len", category="largest_content")
            for row in largest_content
        ],
        "largest_tool_calls_rows": [
            make_row(row, field="tool_calls", length_key="tool_calls_len", category="largest_tool_calls")
            for row in largest_tool_calls
        ],
        "suspicious_data_uri_content_rows": [
            make_row(row, field="content", length_key="content_len", category="data_uri_base64")
            for row in data_uri_content
        ],
        "suspicious_data_uri_tool_calls_rows": [
            make_row(row, field="tool_calls", length_key="tool_calls_len", category="data_uri_base64")
            for row in data_uri_tool_calls
        ],
        "suspicious_base64_like_rows": generic_rows,
        "quarantined_assistant_rows": quarantined_assistant_rows,
        "suspicious_repetitive_assistant_rows": suspicious_repetitive_assistant_rows,
        "heartbeat_noise_rows": heartbeat_noise_rows,
    }

def externalized_payload_stats(config, hermes_home: str = "") -> dict[str, Any]:
    from .externalize import get_large_output_storage_dir

    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    count = 0
    total_bytes = 0
    total_chars = 0
    latest_path = ""
    latest_mtime = 0.0
    if storage_dir.exists() and storage_dir.is_dir():
        for path in storage_dir.glob("*.json"):
            if not path.is_file():
                continue
            count += 1
            try:
                stat = path.stat()
                total_bytes += int(stat.st_size)
                if stat.st_mtime > latest_mtime:
                    latest_mtime = stat.st_mtime
                    latest_path = str(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                total_chars += int(payload.get("content_chars") or len(payload.get("content", "") or ""))
            except Exception:
                continue
    return {
        "externalized_payload_dir": str(storage_dir),
        "externalized_payload_count": count,
        "externalized_payload_bytes": total_bytes,
        "externalized_payload_chars": total_chars,
        "latest_externalized_payload_path": latest_path,
        "latest_externalized_payload_mtime": latest_mtime,
    }

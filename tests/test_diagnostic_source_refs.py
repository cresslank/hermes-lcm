"""Synthetic source-container regressions; never load production transcripts."""
import difflib
import json
import sqlite3

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.ingest_protection import (
    _refs_for_externalized_integrity_scan,
    extract_all_externalized_payload_refs,
    scan_externalized_payload_integrity,
)


KINDS = [
    "Externalized LCM ingest payload: kind=ingest_payload; field=content",
    "Externalized payload: kind=raw_payload; role=assistant",
    "Externalized tool output: tool_call_id=synthetic-call",
    "GC'd externalized payload: kind=raw_payload; role=assistant",
    "GC'd externalized tool output: tool_call_id=synthetic-call",
]


def placeholder(ref="a18d46.json", kind=KINDS[1]):
    return f"[{kind}; chars=12; bytes=12; ref={ref}]"


def source_text(token):
    # Whole, embedded, escaped nested, and split-string source fixtures. Names
    # intentionally do not have example/fake/fixture prefixes.
    return "\n".join([
        "def test_synthetic():",
        f"    value = {token!r}",
        f"    embedded = {'prefix ' + token + ' suffix'!r}",
        f"    nested = {json.dumps({'text': token})!r}",
        '    split = ("' + token.split("; chars=")[0] + '; "',
        '             "chars=12; bytes=12; ref=a18d46.json]")',
    ]) + "\n"


def diff_text(source):
    return "".join(difflib.unified_diff([], source.splitlines(keepends=True), fromfile="a/tests/synthetic.py", tofile="b/tests/synthetic.py"))


def patch_result(source):
    return {"success": True, "diff": diff_text(source), "files_modified": ["tests/synthetic.py"],
            "resolved_path": "/synthetic/tests/synthetic.py", "lint": {"status": "ok", "output": ""},
            "_warning": "synthetic partial-read warning"}


def read_result(source):
    return {"content": "\n".join(f"{i}|{line}" for i, line in enumerate(source.splitlines(), 201)),
            "total_lines": 300, "file_size": len(source), "truncated": True,
            "hint": "Use offset=250", "is_binary": False, "is_image": False}


def terminal_result(output):
    return {"output": output, "exit_code": 1, "error": None, "approval": "synthetic approval"}


def pytest_output(source):
    return "\n".join([
        "================ FAILURES ================", "_________ test_synthetic _________", "",
        *["    " + line for line in source.splitlines()],
        ">       assert False", "E       AssertionError", "tests/synthetic.py:14: AssertionError",
        "========== short test summary info ==========",
    ])


def container(family, source):
    if family == "patch":
        return "patch", patch_result(source)
    if family == "read_file":
        return "read_file", read_result(source)
    if family == "terminal-diff":
        return "terminal", terminal_result(diff_text(source))
    if family == "pytest":
        return "terminal", terminal_result(pytest_output(source))
    if family == "numbered":
        return "terminal", terminal_result("\n".join(f"{i}: {line}" for i, line in enumerate(source.splitlines(), 101)))
    raise AssertionError(family)


FAMILIES = ["patch", "read_file", "terminal-diff", "pytest", "numbered"]


def scan(tool_name, value, **kw):
    text = value if isinstance(value, str) else json.dumps(value)
    return set(_refs_for_externalized_integrity_scan(text, role="tool", field="content", tool_name=tool_name, **kw))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("family", FAMILIES)
def test_known_source_families_do_not_own_quoted_or_split_fixtures(family, kind):
    tool, result = container(family, source_text(placeholder(kind=kind)))
    assert scan(tool, result) == set()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", KINDS)
def test_source_and_real_embedded_ref_on_same_line_keep_real_ref(family, kind):
    quoted = placeholder()
    owner = placeholder("b27f53.json", kind)
    tool, result = container(family, f"def test_synthetic():\n    value = {quoted!r}; owner: {owner}\n")
    assert scan(tool, result) == {"b27f53.json"}


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("wrapped", ["expand", "terminal", "rows", "rows-no-tool", "row-lines"])
def test_nested_diagnostic_echo_retains_source_provenance(family, wrapped):
    tool, result = container(family, source_text(placeholder()))
    owner = placeholder("c36e62.json")
    if wrapped == "expand":
        # The real raw-message lcm_expand response omits original tool_name.
        wrapped_tool = "lcm_expand"
        value = {"source_type": "raw_message", "store_id": 7, "role": "tool", "content": json.dumps(result),
                 "externalized_refs": ["a18d46.json"], "externalized_note": owner}
    elif wrapped == "terminal":
        wrapped_tool = "terminal"
        value = terminal_result(json.dumps(result))
        value["error"] = owner
    elif wrapped in {"rows", "rows-no-tool"}:
        wrapped_tool = "terminal"
        value = terminal_result(json.dumps([
            {"store_id": 7, "role": "tool", "tool_name": tool if wrapped == "rows" else None, "content": json.dumps(result)},
            {"store_id": 8, "role": "assistant", "content": owner},
        ]))
    else:
        wrapped_tool = "terminal"
        value = terminal_result("ROW 7\n" + json.dumps(result) + "\nPATH synthetic\n" + owner)
    assert scan(wrapped_tool, value) == {"c36e62.json"}


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mutation", ["extra-key", "wrong-type", "malformed-json", "wrong-tool", "wrong-role"])
def test_uncertain_source_containers_fail_closed(family, mutation):
    tool, value = container(family, source_text(placeholder()))
    if mutation == "extra-key":
        value["provider_extension"] = True
    elif mutation == "wrong-type":
        key = {"patch": "success", "read_file": "total_lines"}.get(tool, "exit_code")
        value[key] = "unknown"
    elif mutation == "malformed-json":
        value = json.dumps(value)[:-1]
    elif mutation == "wrong-tool":
        tool = "unknown_tool"
    else:
        assert "a18d46.json" in _refs_for_externalized_integrity_scan(json.dumps(value), role="assistant", field="content", tool_name=tool)
        return
    assert "a18d46.json" in scan(tool, value)


@pytest.mark.parametrize("field", ["error", "_warning", "lint", "resolved_path", "files_modified"])
def test_patch_non_source_fields_keep_genuine_refs(field):
    value = patch_result(source_text(placeholder()))
    token = placeholder("d45c71.json")
    value[field] = {"status": "error", "output": token} if field == "lint" else [token] if field == "files_modified" else token
    assert scan("patch", value) == {"d45c71.json"}


@pytest.mark.parametrize("tool,field", [("terminal", "error"), ("terminal", "approval"), ("read_file", "hint"), ("read_file", "error")])
def test_other_non_source_fields_keep_genuine_refs(tool, field):
    value = terminal_result(pytest_output(source_text(placeholder()))) if tool == "terminal" else read_result(source_text(placeholder()))
    value[field] = placeholder("d45c71.json")
    assert scan(tool, value) == {"d45c71.json"}


@pytest.mark.parametrize("tool", ["patch", "read_file", "terminal", "lcm_expand", "unknown"])
@pytest.mark.parametrize("kind", KINDS)
def test_whole_field_and_ordinary_quoted_refs_remain_owning(tool, kind):
    token = placeholder(kind=kind)
    for value in (token, f"ordinary output: {token}", f"ordinary output: {token!r}"):
        assert scan(tool, value) == {"a18d46.json"}
        assert extract_all_externalized_payload_refs(value) == ["a18d46.json"]


@pytest.mark.parametrize("text", [
    '+ value = "' + placeholder() + '"',
    '--- a/test.py\n+++ b/test.py\n@@ -0,0 +1,2 @@\n+ value = "' + placeholder() + '"',
    '    def test_synthetic():\n        value = "' + placeholder() + '"',
    'E       AssertionError: "' + placeholder() + '"',
])
def test_incomplete_source_context_is_not_suppressed(text):
    assert scan("terminal", terminal_result(text)) == {"a18d46.json"}


def test_hunk_and_pytest_source_boundaries_preserve_later_output():
    for output in (diff_text(source_text(placeholder())), pytest_output(source_text(placeholder()))):
        value = terminal_result(output + '\nordinary output: "' + placeholder("e54b80.json") + '"')
        assert scan("terminal", value) == {"e54b80.json"}


def test_unknown_nested_envelope_fails_closed():
    inner = patch_result(source_text(placeholder()))
    inner["unknown"] = "data"
    assert "a18d46.json" in scan("terminal", terminal_result(json.dumps(inner)))
    expanded = {"source_type": "raw_message", "store_id": 7, "role": "tool", "content": json.dumps(inner)}
    assert "a18d46.json" in scan("lcm_expand", expanded)


def test_excessive_nested_diagnostics_fail_closed_without_crashing():
    value = patch_result(source_text(placeholder()))
    for _ in range(10):
        value = terminal_result(json.dumps(value))
    assert "a18d46.json" in scan("terminal", value)


@pytest.mark.parametrize("key,value", [("provider_extension", True), ("store_id", True), ("content_chars", "wrong"), ("role", None)])
def test_unknown_or_malformed_recall_envelopes_fail_closed(key, value):
    result = {"source_type": "raw_message", "store_id": 7, "role": "tool", "content": json.dumps(patch_result(source_text(placeholder())))}
    result[key] = value
    assert "a18d46.json" in scan("lcm_expand", result)


@pytest.mark.parametrize("family", FAMILIES)
def test_duplicate_source_envelope_keys_fail_closed(family):
    tool, result = container(family, source_text(placeholder()))
    key = {"patch": "diff", "read_file": "content"}.get(tool, "output")
    serialized = json.dumps(result)[:-1] + ', ' + json.dumps(key) + ': "replacement"}'
    assert "a18d46.json" in scan(tool, serialized)
    assert "a18d46.json" in scan("terminal", terminal_result(serialized))


@pytest.mark.parametrize("ref", ["9c213feef51a.json", "other-result.json", "session_123.json"])
def test_source_classification_does_not_depend_on_ref_filename(ref):
    token = placeholder(ref)
    assert scan("patch", patch_result(f"value = {token!r}\n")) == set()
    assert scan("patch", patch_result(f"value = {token}\n")) == {ref}


def test_nested_recall_envelopes_keep_source_and_outer_owned_refs():
    value = patch_result(source_text(placeholder()))
    for index in range(2):
        value = {"source_type": "raw_message", "store_id": index + 1, "role": "tool", "content": json.dumps(value)}
    value["externalized_note"] = placeholder("true-owner.json")
    assert scan("lcm_expand", value) == {"true-owner.json"}
    assert scan("terminal", terminal_result(json.dumps(value))) == {"true-owner.json"}


def test_diagnostic_row_preserves_tool_call_source_and_real_ownership():
    def row(new_source):
        return {"store_id": 7, "role": "assistant", "content": "", "tool_calls": json.dumps([
            {"type": "function", "function": {"name": "patch", "arguments": json.dumps({
                "mode": "replace", "path": "synthetic.py", "old_string": "old", "new_string": new_source,
            })}},
        ])}
    source = "value = " + repr(placeholder())
    owner = placeholder("owner-tool-call.json", KINDS[0])
    value = terminal_result(json.dumps([row(source), row(owner)]))
    assert scan("terminal", value) == {"owner-tool-call.json"}
    malformed = row(source)
    malformed["unknown"] = "extension"
    assert "a18d46.json" in scan("terminal", terminal_result(json.dumps([malformed])))


def test_invalid_large_hunk_counts_fail_closed():
    output = '--- a/test.py\n+++ b/test.py\n@@ -0,0 +1,' + '9' * 5000 + ' @@\n+value = ' + repr(placeholder())
    assert scan("terminal", terminal_result(output)) == {"a18d46.json"}


def test_integrity_scan_reports_only_true_missing_owner_with_source_echoes(tmp_path):
    # Minimal synthetic store; no payload contents are opened or needed.
    config = LCMConfig(large_output_externalization_path=str(tmp_path / "externalized"))
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE messages (store_id INTEGER, session_id TEXT, source TEXT, role TEXT, tool_name TEXT, content TEXT, tool_calls TEXT)")
        for index, family in enumerate(FAMILIES, 1):
            tool, value = container(family, source_text(placeholder()))
            conn.execute("INSERT INTO messages VALUES (?, 'synthetic', 'cli', 'tool', ?, ?, NULL)", (index, tool, json.dumps(value)))
        conn.execute("INSERT INTO messages VALUES (99, 'synthetic', 'cli', 'assistant', NULL, ?, NULL)", (placeholder("f63a99.json", KINDS[0]),))
        result = scan_externalized_payload_integrity(conn, config, hermes_home=str(tmp_path))
        assert result["externalized_payload_refs_total"] == 1
        assert result["externalized_payload_refs_missing"] == 1
        assert result["missing_externalized_payload_refs"] == [{
            "store_id": 99, "session_id": "synthetic", "source": "cli", "role": "assistant", "field": "content", "externalized_ref": "f63a99.json",
        }]
    finally:
        conn.close()


@pytest.mark.parametrize("include_missing_owner", [False, True])
def test_both_doctor_surfaces_use_source_provenance_without_mutating_rows(tmp_path, include_missing_owner):
    from hermes_lcm.command import handle_lcm_command
    from hermes_lcm.engine import LCMEngine
    from hermes_lcm.tools import lcm_doctor

    engine = LCMEngine(LCMConfig(
        database_path=str(tmp_path / "synthetic.db"), sqlite_journal_mode="delete",
        large_output_externalization_path=str(tmp_path / "externalized"),
    ), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("diagnostic-synthetic", platform="cli")
    try:
        for family in FAMILIES:
            tool, value = container(family, source_text(placeholder()))
            engine._store.append("diagnostic-synthetic", {
                "role": "tool", "tool_name": tool, "content": json.dumps(value),
            }, token_estimate=1)
        if include_missing_owner:
            engine._store.append("diagnostic-synthetic", {
                "role": "assistant", "content": placeholder("actual-missing.json"),
            }, token_estimate=1)
        conn = engine._store._conn
        before = [tuple(row) for row in conn.execute("SELECT * FROM messages ORDER BY store_id")]
        report = json.loads(lcm_doctor({}, engine=engine))
        payload = next(check for check in report["checks"] if check["check"] == "payload_storage")
        expected = int(include_missing_owner)
        assert payload["detail"]["externalized_payload_refs_total"] == expected
        assert payload["detail"]["externalized_payload_refs_missing"] == expected
        assert payload["status"] == ("warn" if include_missing_owner else "pass")
        command_report = handle_lcm_command("doctor", engine)
        assert f"externalized_payload_refs_total: {expected}" in command_report
        assert f"externalized_payload_refs_missing: {expected}" in command_report
        assert [tuple(row) for row in conn.execute("SELECT * FROM messages ORDER BY store_id")] == before
    finally:
        engine.shutdown()


# Remaining diagnostic transports: all structures below are synthetic. No
# transcript text, payload contents, observed row IDs, or filename allowlists.
def compact_rtk(source, *, omitted=0, filename="notes/arbitrary.data"):
    body = source.splitlines()
    count = len(body) + omitted
    return "\n".join([
        f" {filename} | {count} +++", f" 1 file changed, {count} insertions(+)",
        "", "Changes:", "", filename, f"@@ -0,0 +1,{count} @@",
        *["+" + line for line in body],
        *([f"  ... ({omitted} additions truncated)"] if omitted else []),
        f"  +{count} -0", "[full diff: rtk git diff --no-compact]",
    ])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("omitted", [0, 4])
def test_rtk_compact_diff_requires_provenance_and_keeps_owners(kind, omitted):
    source = source_text(placeholder(kind=kind)) + "owner: " + placeholder("rtk-owned.json", kind)
    output = compact_rtk(source, omitted=omitted)
    output += "\nordinary: " + repr(placeholder("after-rtk.json", kind))
    assert scan("terminal", terminal_result(output)) == {"rtk-owned.json", "after-rtk.json"}


def test_rtk_multiple_paths_hunks_context_and_removed_lines():
    token = placeholder("same-name.json")
    output = "\n".join([
        " lib/arbitrary.cc | 2 +-", " notes/other file | 1 +",
        " 2 files changed, 2 insertions(+), 1 deletion(-)", "", "Changes:", "",
        "lib/arbitrary.cc", "@@ -10,2 +10,2 @@",
        " context = " + repr(token), "-old = " + repr(token), "+new = " + repr(token),
        "  +1 -1", "", "notes/other file", "@@ -0,0 +1,1 @@",
        "+" + repr(token) + "; owner: " + token,
        "  +1 -0", "[full diff: rtk git diff --no-compact]",
    ])
    assert scan("terminal", terminal_result(output)) == {"same-name.json"}
    assert scan("terminal", terminal_result(output.replace("; owner: " + token, ""))) == set()


@pytest.mark.parametrize("mutation", [
    "footer", "changes", "stats", "unknown-path", "missing-sentinel",
    "wrong-omission", "old-lines-missing", "short-header", "huge-count", "extra-field",
])
def test_rtk_incomplete_or_unknown_compact_context_fails_closed(mutation):
    text = compact_rtk("value = " + repr(placeholder()), omitted=4)
    if mutation == "footer":
        text = text.replace("[full diff: rtk git diff --no-compact]", "")
    elif mutation == "changes":
        text = text.replace("Changes:", "Unknown:")
    elif mutation == "stats":
        text = text.replace("1 file changed", "2 files changed")
    elif mutation == "unknown-path":
        text = text.replace("\nnotes/arbitrary.data\n", "\nother/name\n")
    elif mutation == "missing-sentinel":
        text = text.replace("  ... (4 additions truncated)", "")
    elif mutation == "wrong-omission":
        text = text.replace("4 additions truncated", "3 additions truncated")
    elif mutation == "old-lines-missing":
        text = text.replace("@@ -0,0", "@@ -1,1")
    elif mutation == "short-header":
        text = text.replace("@@ -0,0 +1,5 @@", "@@ +1,5 @@")
    elif mutation == "huge-count":
        text = text.replace("@@ -0,0 +1,5 @@", "@@ -0,0 +1," + "9" * 5000 + " @@")
    value = terminal_result(text)
    if mutation == "extra-field":
        value["extension"] = True
    assert "a18d46.json" in scan("terminal", value)


def excerpt_row(head, tail="", *, content_len=None, tool="patch"):
    return {"store_id": 11, "role": "tool", "source": "synthetic", "tool_name": tool,
            "content_len": len(head) if content_len is None else content_len,
            "content_head": head, "content_tail": tail}


def test_complete_metadata_excerpt_delegates_but_preserves_ordinary_owners():
    complete = json.dumps(patch_result(source_text(placeholder())))
    owner = placeholder("excerpt-owned.json")
    rows = [excerpt_row(complete, complete), excerpt_row(owner, owner),
            excerpt_row('ordinary quoted: ' + repr(owner), content_len=1000)]
    assert scan("terminal", terminal_result(json.dumps(rows))) == {"excerpt-owned.json"}


@pytest.mark.parametrize("kind", KINDS)
def test_truncated_metadata_patch_excerpts_intentionally_remain_flagged(kind):
    complete = json.dumps(patch_result(source_text(placeholder(kind=kind))))
    # Complete outer array/metadata, incomplete inner patch reply. Its unknown
    # trailing keys cannot be validated; we must not infer a trusted envelope.
    row = excerpt_row(complete[:-12], complete[10:], content_len=len(complete))
    assert "a18d46.json" in scan("terminal", terminal_result(json.dumps([row])))


@pytest.mark.parametrize("mutation", ["unknown-field", "bad-length", "bool-length", "bool-id", "wrong-role", "wrong-tool", "duplicate-key", "malformed"])
def test_metadata_excerpt_unknown_or_malformed_fails_closed(mutation):
    complete = json.dumps(patch_result(source_text(placeholder())))
    row = excerpt_row(complete)
    if mutation == "unknown-field":
        row["extension"] = True
    elif mutation == "bad-length":
        row["content_len"] -= 1
    elif mutation == "bool-length":
        row["content_len"] = True
    elif mutation == "bool-id":
        row["store_id"] = True
    elif mutation == "wrong-role":
        row["role"] = "assistant"
    elif mutation == "wrong-tool":
        row["tool_name"] = "unknown"
    encoded = json.dumps([row])
    if mutation == "duplicate-key":
        encoded = encoded[:-2] + ', "content_head": ""}]'
    elif mutation == "malformed":
        encoded = encoded[:-1]
    assert "a18d46.json" in scan("terminal", terminal_result(encoded))


def numbered_slice(source, *, prefix="leading fragment", suffix="trailing fragment"):
    lines = "\n".join(f"  {index}: {line}" for index, line in enumerate(source.splitlines(), 311))
    # Exactly one serialized JSON layer, intentionally incomplete as a whole.
    return prefix + json.dumps("\n" + lines + "\n")[1:-1] + suffix


@pytest.mark.parametrize("header", ["ROW 21 role tool tool terminal", "ROW 22 role assistant tool None"])
@pytest.mark.parametrize("kind", KINDS)
def test_repr_slices_only_suppress_explicit_numbered_source(header, kind):
    token = placeholder(kind=kind)
    source = source_text(token) + "owner: " + placeholder("slice-owned.json", kind)
    fragment = numbered_slice(source, prefix='ref=earlier.json]\\",',
                              suffix='ordinary: ' + repr(placeholder("outside-slice.json", kind)))
    output = header + "\n" + repr(fragment)
    assert scan("terminal", terminal_result(output)) == {"slice-owned.json", "outside-slice.json"}


def test_alternating_repr_rows_keep_same_name_owner_and_unrelated_output():
    token = placeholder()
    first = numbered_slice("value = " + repr(token))
    output = "\n".join([
        "ROW 21 role tool tool terminal", repr(first),
        "ROW 22 role assistant tool None", repr(token),
        "ordinary output: " + repr(placeholder("adjacent.json")),
        repr(numbered_slice("value = " + repr(placeholder("unwrapped.json")))),
    ])
    assert scan("terminal", terminal_result(output)) == {"a18d46.json", "adjacent.json", "unwrapped.json"}


@pytest.mark.parametrize("mutation", [
    "no-header", "unknown-role", "unknown-tool", "extra-header", "missing-quote",
    "expression", "concat", "prefix", "bad-escape", "missing-newline", "unnumbered", "extra-terminal-key",
])
def test_repr_slices_reject_malformed_or_unknown_context(mutation):
    header = "ROW 21 role tool tool terminal"
    fragment = numbered_slice("value = " + repr(placeholder()))
    if mutation == "no-header":
        header = ""
    elif mutation == "unknown-role":
        header = header.replace("role tool", "role user")
    elif mutation == "unknown-tool":
        header = header.replace("terminal", "read_file")
    elif mutation == "extra-header":
        header += " extra"
    elif mutation == "bad-escape":
        fragment = fragment.replace("311: ", "311: \\q")
    elif mutation == "missing-newline":
        fragment = fragment.removesuffix("\\ntrailing fragment")
    elif mutation == "unnumbered":
        fragment = fragment.replace("311:", "note:")
    encoded = repr(fragment)
    if mutation == "missing-quote":
        encoded = encoded[:-1]
    elif mutation == "expression":
        encoded += ".upper()"
    elif mutation == "concat":
        encoded += " ''"
    elif mutation == "prefix":
        encoded = "r" + encoded
    result = terminal_result(header + "\n" + encoded)
    if mutation == "extra-terminal-key":
        result["unknown"] = "field"
    assert "a18d46.json" in scan("terminal", result)


def test_repr_rows_do_not_drop_plain_or_whole_field_owners():
    token = placeholder()
    for fragment in (token, repr(token), "plain: " + repr(token)):
        assert scan("terminal", terminal_result("ROW 21 role tool tool terminal\n" + repr(fragment))) == {"a18d46.json"}


def test_repr_path_output_lines_mask_source_literals_only():
    token = placeholder()
    source = "401: " + repr(token) + "; owner: " + placeholder("path-owner.json")
    output = f"ROW 31 content\nPATH $.output len {len(source)}\n{source!r}"
    assert scan("terminal", terminal_result(output)) == {"path-owner.json"}
    assert "a18d46.json" in scan("terminal", terminal_result(output.replace("$.output", "$.unknown")))
    assert "a18d46.json" in scan("terminal", terminal_result(output.replace(str(len(source)), "1")))


def diagnostic_arguments(new_source, **overrides):
    args = {"mode": "replace", "path": "src/arbitrary.data", "old_string": "old", "new_string": new_source}
    args.update(overrides)
    return json.dumps(args)


def argument_row(encoded):
    return f"ROW 31 tool_calls\nPATH $[0].function.arguments len {len(encoded)}\n{encoded!r}"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("owner_field", ["path", "old_string", "new_string"])
def test_repr_path_patch_arguments_keep_genuine_owning_arguments(kind, owner_field):
    owner = placeholder("argument-owner.json", kind)
    args = {owner_field: owner} if owner_field != "new_string" else {}
    source = source_text(placeholder(kind=kind))
    if owner_field == "new_string":
        source += "\nowner: " + owner
    encoded = diagnostic_arguments(source, **args)
    assert scan("terminal", terminal_result(argument_row(encoded))) == {"argument-owner.json"}


@pytest.mark.parametrize("mutation", ["extra-key", "duplicate-key", "wrong-mode", "malformed", "bad-path", "bad-row", "bad-length", "bool-source", "args-only"])
def test_repr_path_patch_arguments_fail_closed(mutation):
    encoded = diagnostic_arguments(source_text(placeholder()))
    if mutation == "extra-key":
        encoded = encoded[:-1] + ', "unknown": true}'
    elif mutation == "duplicate-key":
        encoded = encoded[:-1] + ', "new_string": ""}'
    elif mutation == "wrong-mode":
        encoded = encoded.replace('"replace"', '"patch"')
    elif mutation == "malformed":
        encoded = encoded[:-1]
    elif mutation == "bool-source":
        encoded = encoded.replace('"old"', 'true')
    output = argument_row(encoded)
    if mutation == "bad-path":
        output = output.replace("$[0].function.arguments", "$.arguments")
    elif mutation == "bad-row":
        output = output.replace("tool_calls", "content")
    elif mutation == "bad-length":
        output = output.replace(f"len {len(encoded)}", "len 1")
    elif mutation == "args-only":
        output = repr(encoded)
    assert "a18d46.json" in scan("terminal", terminal_result(output))


def test_new_repr_wrappers_keep_depth_and_size_bounds():
    output = argument_row(diagnostic_arguments(source_text(placeholder())))
    result = terminal_result(output)
    for _ in range(8):
        result = terminal_result(json.dumps(result))
    assert "a18d46.json" in scan("terminal", result)
    assert "a18d46.json" in scan("terminal", terminal_result(" " * 2_000_001 + output))


@pytest.mark.parametrize("mutation", ["unknown-field", "duplicate-key", "malformed-closed"])
def test_complete_repr_json_does_not_bypass_exact_envelope_validation(mutation):
    source = 'before\n 401: "' + placeholder() + '"\nafter'
    encoded = json.dumps(terminal_result(source))
    if mutation == "unknown-field":
        encoded = encoded[:-1] + ', "unknown": true}'
    elif mutation == "duplicate-key":
        encoded = encoded[:-1] + ', "output": "replacement"}'
    else:
        encoded = encoded[:-1] + ',}'
    text = "ROW 41 role tool tool terminal\n" + repr(encoded)
    assert "a18d46.json" in scan("terminal", terminal_result(text))


def test_complete_repr_json_can_reuse_recognized_source_envelope():
    encoded = json.dumps(terminal_result('before\n 401: "' + placeholder() + '"\nafter'))
    text = "ROW 41 role tool tool terminal\n" + repr(encoded)
    assert scan("terminal", terminal_result(text)) == set()


@pytest.mark.parametrize("kind", KINDS)
def test_repr_slice_source_and_same_name_owner_on_one_line(kind):
    token = placeholder("ordinary-name.json", kind)
    source = repr(token) + "; owner: " + token
    text = "ROW 41 role tool tool terminal\n" + repr(numbered_slice(source))
    assert scan("terminal", terminal_result(text)) == {"ordinary-name.json"}


def test_repr_slice_keeps_placeholder_crossing_known_and_unknown_regions():
    token = placeholder("boundary-owner.json")
    start, end = token.split("; chars=")
    # An owning match begins outside the numbered source region. The local
    # recognizer cannot discard it just because its end lies in quoted source.
    fragment = start + numbered_slice('value = "; chars=' + end + '"')
    text = "ROW 41 role tool tool terminal\n" + repr(fragment)
    assert scan("terminal", terminal_result(text)) == {"boundary-owner.json"}


def test_repr_slice_never_decodes_an_arbitrary_extra_escape_layer():
    fragment = json.dumps(numbered_slice("value = " + repr(placeholder())))[1:-1]
    text = "ROW 41 role tool tool terminal\n" + repr(fragment)
    assert "a18d46.json" in scan("terminal", terminal_result(text))


def test_rtk_omission_accounts_for_trailing_unchanged_context():
    source = "value = " + repr(placeholder()) + "; owner: " + placeholder("rtk-real.json")
    output = compact_rtk(source, omitted=4).replace("@@ -0,0 +1,5 @@", "@@ -20,3 +20,8 @@")
    assert scan("terminal", terminal_result(output)) == {"rtk-real.json"}
    assert "a18d46.json" in scan("terminal", terminal_result(output.replace("+20,8", "+20,9")))


@pytest.mark.parametrize("transport", ["json", "repr"])
@pytest.mark.parametrize("include_owner", [False, True])
def test_integrity_resolves_verified_copied_excerpt_against_original(tmp_path, transport, include_owner):
    fake = placeholder("quoted-example.json")
    real = placeholder("real-owned.json")
    original = json.dumps(terminal_result("71: value = " + repr(fake) + ("\nowner: " + real if include_owner else "")))
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, source TEXT, role TEXT, tool_name TEXT, content TEXT, tool_calls TEXT)")
        conn.execute("INSERT INTO messages VALUES (11, 'synthetic', 'synthetic', 'tool', 'terminal', ?, NULL)", (original,))
        if transport == "json":
            output = json.dumps([excerpt_row(original[:-7], original[12:-1], content_len=len(original), tool="terminal")])
        else:
            output = "ROW 11 role tool tool terminal\n" + repr(original[10:-3])
        conn.execute("INSERT INTO messages VALUES (20, 'synthetic', 'synthetic', 'tool', 'terminal', ?, NULL)", (json.dumps(terminal_result(output)),))
        before = list(conn.execute("SELECT * FROM messages ORDER BY store_id"))
        result = scan_externalized_payload_integrity(conn, LCMConfig(large_output_externalization_path=str(tmp_path)), hermes_home=str(tmp_path))
        assert result["externalized_payload_refs_missing"] == int(include_owner)
        assert [item["externalized_ref"] for item in result["missing_externalized_payload_refs"]] == (["real-owned.json"] if include_owner else [])
        assert list(conn.execute("SELECT * FROM messages ORDER BY store_id")) == before
    finally:
        conn.close()


@pytest.mark.parametrize("transport", ["json", "repr"])
@pytest.mark.parametrize("mutation", ["missing-source", "future-source", "self-source", "changed-source", "wrong-role", "unknown-envelope"])
def test_copied_excerpt_requires_matching_prior_source_or_fails_closed(tmp_path, transport, mutation):
    token = placeholder("unverified.json")
    original = json.dumps(terminal_result("71: value = " + repr(token)))
    source_id = 21 if mutation == "future-source" else 20 if mutation == "self-source" else 11
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, source TEXT, role TEXT, tool_name TEXT, content TEXT, tool_calls TEXT)")
        if mutation not in {"missing-source", "self-source"}:
            conn.execute("INSERT INTO messages VALUES (?, 'synthetic', 'synthetic', 'tool', 'terminal', ?, NULL)", (source_id, original.replace("unverified.json", "different.json") if mutation == "changed-source" else original))
        if transport == "json":
            item = excerpt_row(original[:-7], original[12:-1], content_len=len(original), tool="terminal")
            item["store_id"] = source_id
            if mutation == "wrong-role":
                item["role"] = "assistant"
            output = json.dumps([item])
        else:
            output = f"ROW {source_id} role {'assistant' if mutation == 'wrong-role' else 'tool'} tool terminal\n" + repr(original[10:-3])
        outer = terminal_result(output)
        if mutation == "unknown-envelope":
            outer["unknown"] = True
        conn.execute("INSERT INTO messages VALUES (20, 'synthetic', 'synthetic', 'tool', 'terminal', ?, NULL)", (json.dumps(outer),))
        result = scan_externalized_payload_integrity(conn, LCMConfig(large_output_externalization_path=str(tmp_path)), hermes_home=str(tmp_path))
        assert "unverified.json" in [item["externalized_ref"] for item in result["missing_externalized_payload_refs"]]
    finally:
        conn.close()


def test_verified_excerpt_preserves_non_source_outer_reference(tmp_path):
    original = json.dumps(terminal_result("1: " + repr(placeholder("example.json"))))
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, source TEXT, role TEXT, tool_name TEXT, content TEXT, tool_calls TEXT)")
        conn.execute("INSERT INTO messages VALUES (11, 'synthetic', 'synthetic', 'tool', 'terminal', ?, NULL)", (original,))
        outer = terminal_result("ROW 11 role tool tool terminal\n" + repr(original[10:-3]))
        outer["error"] = placeholder("error-owner.json")
        conn.execute("INSERT INTO messages VALUES (20, 'synthetic', 'synthetic', 'tool', 'terminal', ?, NULL)", (json.dumps(outer),))
        result = scan_externalized_payload_integrity(conn, LCMConfig(large_output_externalization_path=str(tmp_path)), hermes_home=str(tmp_path))
        assert [item["externalized_ref"] for item in result["missing_externalized_payload_refs"]] == ["error-owner.json"]
    finally:
        conn.close()

"""Test configuration and process-wide filesystem isolation.

The environment and SQLite guard are installed before any plugin module is
imported.  Tests must never resolve implicit Hermes defaults against a user's
real home directory.
"""

import os
import tempfile

# Establish a process-level sandbox before importing pathlib, pytest, or any
# repository module.  Per-test homes below prevent state sharing inside it.
_INHERITED_HOME = os.environ.get("HOME", "")
_TEST_SANDBOX = tempfile.TemporaryDirectory(prefix="hermes-lcm-pytest-")
_TEST_HOME = os.path.join(_TEST_SANDBOX.name, "home")
_TEST_HERMES_HOME = os.path.join(_TEST_HOME, ".hermes")
os.makedirs(_TEST_HERMES_HOME, exist_ok=True)

# A safety-conscious caller can provide the exact host roots to deny.  For a
# normal direct pytest invocation, protect the inherited user's Hermes home.
if "HERMES_TEST_DENY_SQLITE_ROOTS" not in os.environ and _INHERITED_HOME:
    os.environ["HERMES_TEST_DENY_SQLITE_ROOTS"] = os.path.join(_INHERITED_HOME, ".hermes")
os.environ["HOME"] = _TEST_HOME
os.environ["HERMES_HOME"] = _TEST_HERMES_HOME

import importlib
import sqlite3
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest


class UnsafeSQLitePathError(RuntimeError):
    """Raised before SQLite can open a database under a protected live root."""


# Always protect the path involved in the original incident, even when a test
# runner supplies an already-sandboxed HOME.  Extra roots are pathsep-delimited.
_DENIED_SQLITE_ROOTS = tuple(
    Path(value).expanduser().resolve(strict=False)
    for value in {
        "/Users/cresslank/.hermes",
        *filter(None, os.environ.get("HERMES_TEST_DENY_SQLITE_ROOTS", "").split(os.pathsep)),
    }
)
_SQLITE_OPEN_AUDIT: set[str] = set()
_SQLITE_DENIAL_AUDIT: list[str] = []
_ORIGINAL_SQLITE_CONNECT = sqlite3.connect


def _sqlite_target_path(database) -> Path | None:
    """Return a resolved filesystem target, or None for in-memory databases."""
    try:
        value = os.fsdecode(os.fspath(database))
    except (TypeError, UnicodeError) as exc:
        raise UnsafeSQLitePathError(
            f"pytest SQLite guard cannot inspect database target {database!r}"
        ) from exc

    if value == ":memory:":
        return None

    if value.startswith("file:"):
        parsed = urlsplit(value)
        query = parse_qs(parsed.query)
        uri_path = unquote(parsed.path)
        if uri_path == ":memory:" or query.get("mode") == ["memory"]:
            return None
        if parsed.netloc not in ("", "localhost"):
            raise UnsafeSQLitePathError(
                f"pytest SQLite guard rejects non-local file URI {value!r}"
            )
        value = uri_path

    # sqlite3 accepts an empty filename as a temporary on-disk database whose
    # location is managed by SQLite; it cannot name a protected live file.
    if not value:
        return None
    return Path(value).expanduser().resolve(strict=False)


def _assert_safe_sqlite_target(database) -> None:
    target = _sqlite_target_path(database)
    if target is None:
        return
    rendered = str(target)
    _SQLITE_OPEN_AUDIT.add(rendered)
    if any(target == root or target.is_relative_to(root) for root in _DENIED_SQLITE_ROOTS):
        _SQLITE_DENIAL_AUDIT.append(rendered)
        raise UnsafeSQLitePathError(
            f"pytest blocked SQLite access beneath protected Hermes home: {rendered}"
        )


def _guarded_sqlite_connect(database, *args, **kwargs):
    _assert_safe_sqlite_target(database)
    return _ORIGINAL_SQLITE_CONNECT(database, *args, **kwargs)


def _sqlite_audit_hook(event, args):
    # This also catches code that retained sqlite3.connect before our wrapper.
    if event == "sqlite3.connect" and args:
        _assert_safe_sqlite_target(args[0])


sqlite3.connect = _guarded_sqlite_connect
sys.addaudithook(_sqlite_audit_hook)


@pytest.fixture(autouse=True)
def _isolated_hermes_environment(tmp_path_factory, monkeypatch):
    """Give every test its own HOME and HERMES_HOME by default."""
    # Use a sibling of pytest's per-test tmp_path: populating tmp_path itself
    # breaks tests that intentionally require an empty output directory.
    environment_root = tmp_path_factory.mktemp("hermes-env")
    home = environment_root / "home"
    hermes_home = home / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(
        "SQLite isolation: "
        f"{len(_SQLITE_OPEN_AUDIT)} resolved filesystem target(s) checked; "
        f"{len(_SQLITE_DENIAL_AUDIT)} protected access attempt(s) denied before connect"
    )


# Make the repo root importable (for agent.context_engine etc.)
repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Register the plugin directory as a proper package.
plugin_dir = Path(__file__).resolve().parent.parent
pkg_name = "hermes_lcm"

if pkg_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(plugin_dir / "__init__.py"),
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(plugin_dir)]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod
    # Don't exec the module (it tries to register with ctx).
    # Just make submodules importable.

    for py_file in plugin_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        sub_name = f"{pkg_name}.{py_file.stem}"
        if sub_name not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(
                sub_name,
                str(py_file),
                submodule_search_locations=[],
            )
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sub_mod.__package__ = pkg_name
            sys.modules[sub_name] = sub_mod
            setattr(mod, py_file.stem, sub_mod)
            try:
                sub_spec.loader.exec_module(sub_mod)
            except Exception:
                pass  # some modules may fail (e.g. engine needs agent)

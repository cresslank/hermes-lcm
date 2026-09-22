"""Fresh Python probes with storage isolation and the suite's actual SQLite guard.

Use the same isolated interpreter (including its installed safety preload), but
do not inherit arbitrary LCM, interpreter-search-path, provider, profile, database,
or config overrides. The bootstrap runs the same conftest guard before any plugin
import; its temporary HOME is under TMPDIR.
"""
import os
from pathlib import Path
import subprocess
import sys


def isolated_python_env(root: Path) -> dict[str, str]:
    root = root.resolve()
    for directory in (root / "home", root / "hermes-home", root / "tmp"):
        directory.mkdir(parents=True, exist_ok=True)
    # Preserve the sandbox boundary, not provider credentials/configuration.
    allowed = (
        "PATH", "LANG", "LC_ALL", "SYSTEMROOT",
        "HERMES_TEST_DENY_SQLITE_ROOTS", "LCM_AUDIT_SANDBOX",
    )
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update({
        "HOME": str(root / "home"),
        "HERMES_HOME": str(root / "hermes-home"),
        "TMPDIR": str(root / "tmp"),
        "LCM_DATABASE_PATH": str(root / "lcm.db"),
        "LCM_EXTRACTION_OUTPUT_PATH": str(root / "extractions"),
        "LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH": str(root / "externalized"),
        "LCM_SQLITE_JOURNAL_MODE": "delete",
        "LCM_EMBEDDINGS_ENABLED": "false",
        "LCM_PREANSWER_EVIDENCE_ENABLED": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    return env


def run_isolated_python(script: str, root: Path, *args: str | Path, timeout: int = 30):
    conftest = Path(__file__).resolve().with_name("conftest.py")
    bootstrap = f'''import sys
# Install network denial before conftest or plugin imports. Interpreter-local
# sitecustomize safety preloads have already run and remain in force.
def _deny_test_network(event, args):
    if event in {{"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"}}:
        raise RuntimeError("isolated Python probe forbids network")
sys.addaudithook(_deny_test_network)
import runpy
guard = runpy.run_path({str(conftest)!r}, run_name="_lcm_child_test_guard")
# A plain Python probe has no pytest lifecycle. Register exactly this helper's
# owned root through the SAME scratch-only predicate as the parent pytest run.
from types import SimpleNamespace
from pathlib import Path
session = SimpleNamespace(config=SimpleNamespace(_tmp_path_factory=SimpleNamespace(
    getbasetemp=lambda: Path({str(root.resolve())!r}))))
guard["pytest_sessionstart"](session)
'''
    return subprocess.run(
        [sys.executable, "-c", bootstrap + "\n" + script, *map(str, args)],
        cwd=root,
        env=isolated_python_env(root),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

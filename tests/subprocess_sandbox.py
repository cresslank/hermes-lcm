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
# Match the parent pytest process's checkout import root. CI places its minimal
# agent.context_engine stub here; the isolated cwd and scrubbed PYTHONPATH must
# not make that explicit fixture disappear in a fresh interpreter.
sys.path.insert(0, {str(conftest.parent.parent)!r})
runpy.run_path({str(conftest)!r}, run_name="_lcm_child_test_guard")
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

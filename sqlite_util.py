"""SQLite lock-contention helpers shared by the LCM engine.

Isolated from ``engine.py`` (WS5 seam): lock-contention detection, bounded
``busy_timeout`` changes, and transaction-preserving savepoints are pure SQLite
concerns with no engine state. Callers keep their own policy constants (for
example the session-end timeout budget).
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from typing import Iterator, List


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _sqlite_artifact_error(path: Path, reason: str) -> OSError:
    return OSError(errno.EPERM, f"refusing SQLite artifact {path.name!r}: {reason}", str(path))


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _validate_sqlite_artifact(path: Path, file_stat: os.stat_result) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise _sqlite_artifact_error(path, "not a regular file")
    if file_stat.st_nlink != 1:
        raise _sqlite_artifact_error(path, "link count is not one")


def _require_sqlite_artifact_absent(path: Path, *, directory_fd: int) -> None:
    """Accept a vanished sidecar only while its directory entry stays absent."""
    try:
        current = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    _validate_sqlite_artifact(path, current)
    raise _sqlite_artifact_error(path, "directory entry changed while restricting permissions")


def _open_private_sqlite_directory(path: Path) -> int:
    directory = path.parent
    expected = os.stat(directory, follow_symlinks=False)
    if not stat.S_ISDIR(expected.st_mode):
        raise _sqlite_artifact_error(path, "parent is not a regular directory")
    if expected.st_mode & 0o022:
        raise _sqlite_artifact_error(path, "parent directory is writable by another user")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    directory_fd = os.open(directory, flags)
    opened = os.fstat(directory_fd)
    if not stat.S_ISDIR(opened.st_mode) or not _same_file_identity(expected, opened):
        os.close(directory_fd)
        raise _sqlite_artifact_error(path, "parent directory changed while opening")
    return directory_fd


def _chmod_sqlite_artifact_at(
    path: Path,
    *,
    directory_fd: int,
    allow_sidecar_disappearance: bool = False,
) -> bool:
    """Restrict an existing artifact without ever opening its inode.

    POSIX record locks are process-associated: closing *any* descriptor for an
    inode drops locks that SQLite holds through every other descriptor in the
    process.  Therefore this helper deliberately uses dirfd-relative stat and
    chmod operations rather than the superficially safer open/fchmod pattern.

    The parent directory has already been verified not to be writable by group
    or other.  Pre/post no-follow stats retain regular-file, single-link, and
    identity checks around the unavoidable pathname chmod race.
    """
    try:
        expected = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    _validate_sqlite_artifact(path, expected)

    # Linux versions without fchmodat2 cannot implement AT_SYMLINK_NOFOLLOW
    # for chmod.  The pre-check still rejects an existing symlink and the
    # private parent directory excludes cross-user replacement; identity is
    # revalidated below to detect a same-user race after the operation.
    try:
        if os.chmod in getattr(os, "supports_follow_symlinks", ()):
            os.chmod(
                path.name,
                0o600,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        else:
            os.chmod(path.name, 0o600, dir_fd=directory_fd)
    except FileNotFoundError:
        if not allow_sidecar_disappearance:
            raise
        _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
        return False

    try:
        restricted = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        if allow_sidecar_disappearance:
            _require_sqlite_artifact_absent(path, directory_fd=directory_fd)
            return False
        raise
    _validate_sqlite_artifact(path, restricted)
    if not _same_file_identity(expected, restricted):
        raise _sqlite_artifact_error(path, "directory entry changed while restricting permissions")
    if stat.S_IMODE(restricted.st_mode) != 0o600:
        raise _sqlite_artifact_error(path, "permissions were not restricted to 0600")
    return True


def _create_private_sqlite_main_at(path: Path, *, directory_fd: int) -> bool:
    """Publish a new 0600 main DB without closing a descriptor for its inode.

    A private temporary inode is completely prepared and closed before an
    atomic hard-link publishes it at the database pathname.  Consequently the
    raw descriptor can never coexist with a SQLite connection to that inode.
    If another creator wins, only the descriptor-free existing-file path runs.
    """
    for _attempt in range(8):
        if _chmod_sqlite_artifact_at(path, directory_fd=directory_fd):
            return False

        temporary_name = f".{path.name}.{uuid.uuid4().hex}.create"
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_BINARY", 0)
        fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        won_creation_race = False
        try:
            try:
                created = os.fstat(fd)
                _validate_sqlite_artifact(path, created)
                os.fchmod(fd, 0o600)
                created = os.fstat(fd)
            finally:
                os.close(fd)

            try:
                os.link(
                    temporary_name,
                    path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                pass
            else:
                won_creation_race = True
        finally:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass

        if not won_creation_race:
            continue
        published = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        _validate_sqlite_artifact(path, published)
        if not _same_file_identity(created, published):
            raise _sqlite_artifact_error(path, "directory entry changed while creating")
        if stat.S_IMODE(published.st_mode) != 0o600:
            raise _sqlite_artifact_error(path, "permissions were not restricted to 0600")
        return True
    raise _sqlite_artifact_error(path, "database path changed repeatedly while creating")


def _restrict_existing_sqlite_artifacts(db_path: Path) -> None:
    """Restrict verified, single-link SQLite files without following links."""
    if os.name != "posix":  # pragma: no cover - Windows compatibility fallback
        for artifact in (
            db_path,
            *(db_path.with_name(db_path.name + suffix) for suffix in _SQLITE_SIDECAR_SUFFIXES),
        ):
            try:
                artifact.chmod(0o600)
            except FileNotFoundError:
                continue
        return

    directory_fd = _open_private_sqlite_directory(db_path)
    try:
        _chmod_sqlite_artifact_at(
            db_path,
            directory_fd=directory_fd,
        )
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            _chmod_sqlite_artifact_at(
                db_path.with_name(db_path.name + suffix),
                directory_fd=directory_fd,
                allow_sidecar_disappearance=True,
            )
    finally:
        os.close(directory_fd)


def _prepare_private_sqlite_file(path: Path) -> None:
    """Create or tighten one SQLite file and its existing sidecars safely."""
    if os.name != "posix":  # pragma: no cover - Windows compatibility fallback
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            path.chmod(0o600)
        else:
            try:
                if hasattr(os, "fchmod"):
                    os.fchmod(fd, 0o600)
                else:
                    path.chmod(0o600)
            finally:
                os.close(fd)
        _restrict_existing_sqlite_artifacts(path)
        return

    directory_fd = _open_private_sqlite_directory(path)
    try:
        _create_private_sqlite_main_at(path, directory_fd=directory_fd)
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            _chmod_sqlite_artifact_at(
                path.with_name(path.name + suffix),
                directory_fd=directory_fd,
                allow_sidecar_disappearance=True,
            )
    finally:
        os.close(directory_fd)


def _is_sqlite_locked_error(exc: BaseException) -> bool:
    """Return True when an exception chain represents SQLite lock contention."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if isinstance(current, sqlite3.Error) and "locked" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _sqlite_busy_timeout_ms(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA busy_timeout").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


@contextmanager
def _sqlite_savepoint(conn: sqlite3.Connection) -> Iterator[None]:
    """Isolate helper writes without taking ownership of a caller transaction."""
    # UUID hex contains only identifier-safe characters and keeps every nested
    # helper's SAVEPOINT name unique with a fixed upper bound on name length.
    name = f"lcm_{uuid.uuid4().hex}"
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        try:
            conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
        finally:
            conn.execute(f"RELEASE SAVEPOINT {name}")
        raise
    else:
        conn.execute(f"RELEASE SAVEPOINT {name}")


@contextmanager
def _temporary_sqlite_busy_timeout(
    connections: List[sqlite3.Connection | None],
    timeout_ms: int,
) -> Iterator[None]:
    """Temporarily bound SQLite lock waits for gateway-critical paths."""
    bounded_timeout = max(0, int(timeout_ms))
    originals: list[tuple[sqlite3.Connection, int]] = []
    for conn in connections:
        if conn is None:
            continue
        original = _sqlite_busy_timeout_ms(conn)
        conn.execute(f"PRAGMA busy_timeout={bounded_timeout}")
        originals.append((conn, original))
    try:
        yield
    finally:
        for conn, original in reversed(originals):
            conn.execute(f"PRAGMA busy_timeout={original}")

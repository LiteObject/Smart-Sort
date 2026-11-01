"""Recoverable, no-overwrite file moves and manifest-based undo."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence, TextIO

LOG = logging.getLogger("smart-sort")
STATE_DIR = ".smart_sort"
CLUSTER_DIR_RE = re.compile(r"^Cluster_\d+", re.IGNORECASE)
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass
class Move:
    src: Path
    dst: Path
    cluster: int
    expected_sha256: str | None = None
    expected_size: int | None = None


@dataclass
class ApplyResult:
    moved: int
    failed: int
    manifest: Path | None


@dataclass
class _ManifestMove:
    move_id: str
    src: str
    dst: str
    size: int
    sha256: str
    completed: bool


@dataclass
class _Manifest:
    root: Path
    version: int
    moves: list[_ManifestMove]
    undo_intents: dict[str, str]
    undone: set[str]
    incomplete_tail: int | None
    needs_newline: bool


def _is_link_stat(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def is_link_or_reparse(path: Path) -> bool:
    """Treat inaccessible entries as unsafe, including Windows junctions."""
    try:
        return _is_link_stat(path.lstat())
    except OSError:
        return True


def resolve_root(path: Path) -> Path:
    selected = Path(os.path.abspath(os.fspath(path)))
    for component in reversed((selected, *selected.parents)):
        if _is_link_stat(component.lstat()):
            raise ValueError(f"Selected root passes through a link or reparse point: {component}")
    info = selected.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"Not a directory: {selected}")
    return selected


def assert_safe_path(root: Path, path: Path, *, allow_missing: bool = False) -> Path:
    """Reject paths outside root or through links and Windows reparse points."""
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path is outside the selected root: {candidate}") from exc

    current = root
    for part in (None, *relative.parts):
        if part is not None:
            current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if allow_missing and part is not None:
                continue
            raise
        if _is_link_stat(info):
            raise ValueError(f"Symlink or reparse point is not allowed: {current}")
    return candidate


def safe_relative_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest path must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError(f"Manifest path is not a safe relative path: {value}")
    path = assert_safe_path(root, root / relative, allow_missing=True)
    if path == root:
        raise ValueError("Manifest path cannot be the selected root")
    return path


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(root: Path, directory: Path) -> None:
    assert_safe_path(root, directory, allow_missing=True)
    directory.mkdir(parents=True, exist_ok=True)
    assert_safe_path(root, directory)
    current = root
    for part in directory.relative_to(root).parts:
        _sync_directory(current)
        current = current / part


@contextmanager
def root_lock(root: Path) -> Iterator[None]:
    """Use an OS-held lock so a crashed process cannot leave a stale lock."""
    state_dir = root / STATE_DIR
    _ensure_directory(root, state_dir)
    lock_path = assert_safe_path(root, state_dir / "lock", allow_missing=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"Lock is not a regular file: {lock_path}")
        assert_safe_path(root, lock_path)
        if not os.path.samestat(opened, lock_path.stat()):
            raise ValueError(f"Lock file changed while opening: {lock_path}")
        if opened.st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise OSError(f"Another Smart-Sort operation is active for {root}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                raise OSError(f"Another Smart-Sort operation is active for {root}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_info(root: Path, path: Path) -> os.stat_result:
    assert_safe_path(root, path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Not a regular file: {path}")
    return info


def _same_snapshot(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        os.path.samestat(first, second)
        and first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
    )


def _verify_source(
    root: Path, path: Path, expected: os.stat_result, expected_sha256: str
) -> None:
    before = _file_info(root, path)
    if not _same_snapshot(before, expected) or sha256_file(path) != expected_sha256:
        raise OSError(f"Source changed while preparing the move: {path}")
    if not _same_snapshot(_file_info(root, path), expected):
        raise OSError(f"Source changed while preparing the move: {path}")


def _copy_no_replace(
    root: Path, source: Path, destination: Path, expected: os.stat_result, digest: str
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, stat.S_IMODE(expected.st_mode))
    with os.fdopen(descriptor, "wb") as target:
        created = os.fstat(target.fileno())
        with source.open("rb") as reader:
            if not _same_snapshot(os.fstat(reader.fileno()), expected):
                raise OSError(f"Source changed before copying: {source}")
            shutil.copyfileobj(reader, target, 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())

    copied = _file_info(root, destination)
    if not os.path.samestat(created, copied) or sha256_file(destination) != digest:
        raise OSError(f"Copied file failed checksum verification: {destination}")
    _verify_source(root, source, expected, digest)
    shutil.copystat(source, destination, follow_symlinks=False)
    _sync_directory(destination.parent)
    _verify_source(root, source, expected, digest)
    copied = _file_info(root, destination)
    if not os.path.samestat(created, copied) or sha256_file(destination) != digest:
        raise OSError(f"Copied file changed before source removal: {destination}")
    source.unlink()
    _sync_directory(source.parent)


def _move_no_replace(
    root: Path, source: Path, destination: Path, expected: os.stat_result, digest: str
) -> None:
    _verify_source(root, source, expected, digest)
    assert_safe_path(root, destination, allow_missing=True)
    if os.name == "nt":
        try:
            os.rename(source, destination)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            _copy_no_replace(root, source, destination, expected, digest)
        return

    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as exc:
        unsupported = {errno.EXDEV, errno.EPERM, errno.ENOSYS}
        unsupported.add(getattr(errno, "EOPNOTSUPP", errno.ENOSYS))
        unsupported.add(getattr(errno, "ENOTSUP", errno.ENOSYS))
        if exc.errno not in unsupported:
            raise
        _copy_no_replace(root, source, destination, expected, digest)
        return

    _sync_directory(destination.parent)
    _verify_source(root, source, expected, digest)
    assert_safe_path(root, destination)
    if not os.path.samestat(source.stat(), destination.stat()):
        raise OSError(f"Destination changed before source removal: {destination}")
    source.unlink()
    _sync_directory(source.parent)


def _append_record(handle: TextIO, record: dict) -> None:
    handle.write(json.dumps(record) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def apply_moves(
    root: Path,
    moves: Sequence[Move],
    metadata: dict,
    *,
    preflight: Callable[[], None] | None = None,
) -> ApplyResult:
    root = resolve_root(root)
    moved = 0
    failed = 0
    with root_lock(root):
        if preflight is not None:
            preflight()
        state_dir = root / STATE_DIR
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        manifest = state_dir / f"manifest-{stamp}-{uuid.uuid4().hex}.jsonl"
        assert_safe_path(root, manifest, allow_missing=True)
        with manifest.open("x", encoding="utf-8") as handle:
            _append_record(
                handle,
                {
                    **metadata,
                    "type": "header",
                    "version": 2,
                    "created": datetime.now(timezone.utc).isoformat(),
                    "root": str(root),
                },
            )
            _sync_directory(state_dir)
            for index, move in enumerate(moves):
                try:
                    source = assert_safe_path(root, move.src)
                    destination = assert_safe_path(root, move.dst, allow_missing=True)
                    if source == destination:
                        raise ValueError(f"Source and destination are the same: {source}")
                    info = _file_info(root, source)
                    digest = sha256_file(source)
                    if (
                        move.expected_size is not None
                        and info.st_size != move.expected_size
                    ) or (
                        move.expected_sha256 is not None
                        and digest != move.expected_sha256
                    ):
                        raise OSError(f"Source changed since the plan was reviewed: {source}")
                    _verify_source(root, source, info, digest)
                    _ensure_directory(root, destination.parent)
                    intent = {
                        "type": "intent",
                        "id": str(index),
                        "src": source.relative_to(root).as_posix(),
                        "dst": destination.relative_to(root).as_posix(),
                        "size": info.st_size,
                        "sha256": digest,
                    }
                except (OSError, ValueError) as exc:
                    LOG.error("Could not prepare %s: %s", move.src, exc)
                    failed += 1
                    continue

                try:
                    _append_record(handle, intent)
                except OSError as exc:
                    LOG.error("Could not record move intent for %s: %s", source, exc)
                    failed += 1
                    break

                try:
                    _move_no_replace(root, source, destination, info, digest)
                except (OSError, ValueError) as exc:
                    LOG.error("Could not safely move %s: %s", source, exc)
                    failed += 1
                    continue

                moved += 1
                try:
                    _append_record(handle, {**intent, "type": "move"})
                except OSError as exc:
                    LOG.error(
                        "Moved %s but could not record completion: %s. "
                        "Use the manifest to recover it.",
                        source,
                        exc,
                    )
                    failed += 1
                    break

    return ApplyResult(moved, failed, manifest)


def _validate_move_record(record: dict, move_id: str, completed: bool) -> _ManifestMove:
    src, dst = record.get("src"), record.get("dst")
    size, digest = record.get("size"), record.get("sha256")
    if not isinstance(src, str) or not isinstance(dst, str):
        raise ValueError("Manifest move is missing source or destination")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("Manifest move has an invalid size")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Manifest move has an invalid SHA-256 checksum")
    return _ManifestMove(move_id, src, dst, size, digest, completed)


def _read_manifest(path: Path) -> _Manifest:
    if is_link_or_reparse(path):
        raise ValueError(f"Manifest is missing or is a link: {path}")
    records: list[dict] = []
    incomplete_tail: int | None = None
    needs_newline = False
    with path.open("rb") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                needs_newline = not line.endswith(b"\n")
                continue
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if line.endswith(b"\n"):
                    raise ValueError(f"Invalid manifest record at byte {start}") from exc
                incomplete_tail = start
                LOG.warning("Ignoring incomplete final manifest record at byte %d", start)
                break
            if not isinstance(record, dict):
                raise ValueError(f"Invalid manifest record at byte {start}")
            records.append(record)
            needs_newline = not line.endswith(b"\n")

    if not records or records[0].get("type") != "header":
        raise ValueError(f"{path} is not a Smart-Sort manifest (no header)")
    header = records[0]
    version = header.get("version", 1)
    if isinstance(version, bool) or version not in (1, 2):
        raise ValueError(f"Unsupported manifest version: {version}")
    if incomplete_tail is not None and version == 1:
        raise ValueError("Legacy manifest has an incomplete record; manual recovery is required")
    root_value = header.get("root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("Manifest root must be an absolute path")
    root = resolve_root(Path(root_value))

    moves: list[_ManifestMove] = []
    by_id: dict[str, _ManifestMove] = {}
    undo_intents: dict[str, str] = {}
    undone: set[str] = set()
    for record in records[1:]:
        kind = record.get("type")
        if kind == "intent" and version == 2:
            move_id = record.get("id")
            if not isinstance(move_id, str) or not move_id or move_id in by_id:
                raise ValueError("Manifest contains a missing or duplicate move ID")
            move = _validate_move_record(record, move_id, False)
            moves.append(move)
            by_id[move_id] = move
        elif kind == "move":
            if version == 1:
                move_id = f"legacy-{len(moves)}"
                move = _validate_move_record(record, move_id, True)
                moves.append(move)
                by_id[move_id] = move
            else:
                move_id = record.get("id")
                if not isinstance(move_id, str) or move_id not in by_id:
                    raise ValueError("Manifest completion has no matching move intent")
                completed = _validate_move_record(record, move_id, True)
                previous = by_id[move_id]
                if previous.completed or (
                    completed.src,
                    completed.dst,
                    completed.size,
                    completed.sha256,
                ) != (previous.src, previous.dst, previous.size, previous.sha256):
                    raise ValueError("Manifest completion does not match its move intent")
                previous.completed = True
        elif kind == "undo_intent":
            move_id, target = record.get("move_id"), record.get("target")
            if (
                not isinstance(move_id, str)
                or move_id not in by_id
                or not isinstance(target, str)
                or not target
                or move_id in undo_intents
                or move_id in undone
            ):
                raise ValueError("Manifest contains an invalid undo intent")
            undo_intents[move_id] = target
        elif kind == "undo":
            move_id = record.get("move_id")
            if not isinstance(move_id, str) or move_id not in undo_intents or move_id in undone:
                raise ValueError("Manifest contains an invalid undo completion")
            undone.add(move_id)
        else:
            raise ValueError(f"Unrecognized manifest record: {kind!r}")

    return _Manifest(root, version, moves, undo_intents, undone, incomplete_tail, needs_newline)


def _prepare_manifest_append(path: Path, manifest: _Manifest) -> None:
    if manifest.incomplete_tail is None and not manifest.needs_newline:
        return
    with path.open("r+b") as handle:
        if manifest.incomplete_tail is not None:
            handle.truncate(manifest.incomplete_tail)
        else:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _restored_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    candidate = parent / f"{stem} (restored){suffix}"
    counter = 2
    while candidate.exists():
        candidate = parent / f"{stem} (restored {counter}){suffix}"
        counter += 1
    return candidate


def _prune_empty_cluster_dirs(root: Path, directories: Sequence[Path]) -> None:
    for directory in sorted(set(directories), key=lambda path: len(path.parts), reverse=True):
        current = directory
        while current != root and current.parent != current:
            if not CLUSTER_DIR_RE.match(current.name):
                break
            try:
                assert_safe_path(root, current)
                current.rmdir()
            except (OSError, ValueError):
                break
            current = current.parent


def _undo_moves(
    state: _Manifest, handle: TextIO | None, dry_run: bool
) -> tuple[int, int]:
    restored = 0
    failed = 0
    touched: list[Path] = []
    root = state.root
    for move in reversed(state.moves):
        if move.move_id in state.undone:
            continue
        try:
            source = safe_relative_path(root, move.src)
            destination = safe_relative_path(root, move.dst)
            if source == destination:
                raise ValueError("Manifest source and destination are the same")
            pending_target = state.undo_intents.get(move.move_id)
            if pending_target is not None:
                target = safe_relative_path(root, pending_target)
                if target == destination:
                    raise ValueError("Undo target is the same as the destination")
                if target.exists() and not destination.exists():
                    info = _file_info(root, target)
                    if info.st_size != move.size or sha256_file(target) != move.sha256:
                        raise OSError(f"Recovered file does not match its checksum: {target}")
                    if dry_run:
                        print(f"  already restored {move.dst} -> {pending_target}")
                    else:
                        assert handle is not None
                        _append_record(handle, {"type": "undo", "move_id": move.move_id})
                        touched.append(destination.parent)
                    restored += 1
                    continue
                if target.exists() or not destination.exists():
                    raise OSError(
                        f"Interrupted undo is ambiguous; inspect {destination} and {target}"
                    )
            else:
                if not move.completed:
                    if source.exists() and not destination.exists():
                        LOG.warning("Move did not complete: %s", move.src)
                        continue
                    if source.exists() or not destination.exists():
                        raise OSError(
                            f"Interrupted move is ambiguous; inspect {source} and {destination}"
                        )
                if not destination.exists():
                    raise OSError(f"Missing, cannot undo: {destination}")
                target = _restored_path(source)
                assert_safe_path(root, target, allow_missing=True)

            info = _file_info(root, destination)
            if info.st_size != move.size or sha256_file(destination) != move.sha256:
                raise OSError(f"Checksum changed, refusing to move: {destination}")
            relative_target = target.relative_to(root).as_posix()
            if dry_run:
                print(f"  would restore {move.dst} -> {relative_target}")
                restored += 1
                continue

            assert handle is not None
            _ensure_directory(root, target.parent)
            if pending_target is None:
                _append_record(
                    handle,
                    {"type": "undo_intent", "move_id": move.move_id, "target": relative_target},
                )
            _move_no_replace(root, destination, target, info, move.sha256)
            restored += 1
            _append_record(handle, {"type": "undo", "move_id": move.move_id})
            touched.append(destination.parent)
        except (OSError, ValueError) as exc:
            LOG.error("Could not undo %s: %s", move.dst, exc)
            failed += 1
            if not dry_run:
                break

    if not dry_run:
        _prune_empty_cluster_dirs(root, touched)
    return restored, failed


def undo_manifest(manifest: Path, dry_run: bool = True) -> tuple[int, int]:
    """Undo completed or safely identifiable interrupted moves."""
    manifest = Path(os.path.abspath(os.fspath(manifest)))
    state = _read_manifest(manifest)
    if dry_run:
        return _undo_moves(state, None, True)
    locked_root = state.root
    with root_lock(locked_root):
        state = _read_manifest(manifest)
        if state.root != locked_root:
            raise ValueError("Manifest root changed while acquiring the lock")
        _prepare_manifest_append(manifest, state)
        with manifest.open("a", encoding="utf-8") as handle:
            return _undo_moves(state, handle, False)

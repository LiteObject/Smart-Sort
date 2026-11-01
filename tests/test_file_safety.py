"""Fault-injection and recovery tests for file moves."""

from __future__ import annotations

import errno
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import file_safety as fs  # noqa: E402
import smart_sort as ss  # noqa: E402


class FileSafetyTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_move(self, name="document.txt", content="important original content"):
        source = self.root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content, encoding="utf-8")
        destination = self.root / "Cluster_1" / source.name
        return fs.Move(source, destination, 0)

    def manifests(self):
        return list((self.root / fs.STATE_DIR).glob("manifest-*.jsonl"))

    def records(self, manifest):
        return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

    def cross_device_patch(self):
        primitive = "rename" if os.name == "nt" else "link"
        return mock.patch.object(fs.os, primitive, side_effect=OSError(errno.EXDEV, "cross-device"))

    def test_move_intent_is_durable_before_the_move(self):
        move = self.make_move()
        real_move = fs._move_no_replace

        def verify_intent(*args):
            manifest = self.manifests()[0]
            records = self.records(manifest)
            self.assertEqual([record["type"] for record in records], ["header", "intent"])
            self.assertTrue(move.src.exists())
            self.assertFalse(move.dst.exists())
            return real_move(*args)

        with mock.patch.object(fs, "_move_no_replace", side_effect=verify_intent):
            result = fs.apply_moves(self.root, [move], {})

        self.assertEqual((result.moved, result.failed), (1, 0))
        self.assertEqual([record["type"] for record in self.records(result.manifest)],
                         ["header", "intent", "move"])
        self.assertFalse(move.src.exists())
        self.assertEqual(move.dst.read_text(encoding="utf-8"), "important original content")

    def test_failed_intent_write_never_moves_the_source(self):
        move = self.make_move()
        append = fs._append_record

        def fail_intent(handle, record):
            if record["type"] == "intent":
                raise OSError(errno.ENOSPC, "disk full")
            return append(handle, record)

        with mock.patch.object(fs, "_append_record", side_effect=fail_intent):
            with self.assertLogs("smart-sort", level="ERROR"):
                result = fs.apply_moves(self.root, [move], {})

        self.assertEqual((result.moved, result.failed), (0, 1))
        self.assertTrue(move.src.exists())
        self.assertFalse(move.dst.exists())
        self.assertEqual([record["type"] for record in self.records(result.manifest)], ["header"])
        self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (0, 0))

    def test_completion_write_failure_is_recovered_by_undo(self):
        move = self.make_move()
        append = fs._append_record

        def fail_completion(handle, record):
            if record["type"] == "move":
                raise OSError(errno.ENOSPC, "disk full")
            return append(handle, record)

        with mock.patch.object(fs, "_append_record", side_effect=fail_completion):
            with self.assertLogs("smart-sort", level="ERROR"):
                result = fs.apply_moves(self.root, [move], {})

        self.assertEqual((result.moved, result.failed), (1, 1))
        self.assertFalse(move.src.exists())
        self.assertTrue(move.dst.exists())
        self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (1, 0))
        self.assertEqual(move.src.read_text(encoding="utf-8"), "important original content")
        self.assertFalse(move.dst.exists())
        self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (0, 0))

    def test_interrupt_after_move_is_recovered_and_lock_is_released(self):
        move = self.make_move()
        real_move = fs._move_no_replace

        def interrupt_after_move(*args):
            real_move(*args)
            raise KeyboardInterrupt

        with mock.patch.object(fs, "_move_no_replace", side_effect=interrupt_after_move):
            with self.assertRaises(KeyboardInterrupt):
                fs.apply_moves(self.root, [move], {})

        manifest = self.manifests()[0]
        self.assertFalse(move.src.exists())
        self.assertTrue(move.dst.exists())
        self.assertEqual(fs.undo_manifest(manifest, dry_run=False), (1, 0))
        self.assertTrue(move.src.exists())

    def test_interrupt_before_move_leaves_original_in_place(self):
        move = self.make_move()
        with mock.patch.object(fs, "_move_no_replace", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                fs.apply_moves(self.root, [move], {})

        with self.assertLogs("smart-sort", level="WARNING"):
            result = fs.undo_manifest(self.manifests()[0], dry_run=False)
        self.assertEqual(result, (0, 0))
        self.assertTrue(move.src.exists())
        self.assertFalse(move.dst.exists())

    def test_destination_created_during_move_is_not_overwritten(self):
        move = self.make_move()
        primitive = "rename" if os.name == "nt" else "link"
        actual = getattr(fs.os, primitive)

        def race(source, destination, *args, **kwargs):
            Path(destination).write_text("other person's file", encoding="utf-8")
            return actual(source, destination, *args, **kwargs)

        with mock.patch.object(fs.os, primitive, side_effect=race):
            with self.assertLogs("smart-sort", level="ERROR"):
                result = fs.apply_moves(self.root, [move], {})

        self.assertEqual((result.moved, result.failed), (0, 1))
        self.assertEqual(move.src.read_text(encoding="utf-8"), "important original content")
        self.assertEqual(move.dst.read_text(encoding="utf-8"), "other person's file")
        with self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (0, 1))
        self.assertEqual(move.dst.read_text(encoding="utf-8"), "other person's file")

    def test_cross_device_move_is_copied_verified_and_undoable(self):
        move = self.make_move()
        with self.cross_device_patch():
            result = fs.apply_moves(self.root, [move], {})

        self.assertEqual((result.moved, result.failed), (1, 0))
        self.assertFalse(move.src.exists())
        self.assertEqual(move.dst.read_text(encoding="utf-8"), "important original content")
        self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (1, 0))
        self.assertTrue(move.src.exists())

    def test_disk_full_during_copy_preserves_source_and_reports_ambiguity(self):
        move = self.make_move()

        def fail_copy(reader, writer, length):
            writer.write(reader.read(5))
            raise OSError(errno.ENOSPC, "disk full")

        with self.cross_device_patch(), mock.patch.object(
            fs.shutil, "copyfileobj", side_effect=fail_copy
        ):
            with self.assertLogs("smart-sort", level="ERROR"):
                result = fs.apply_moves(self.root, [move], {})

        self.assertEqual((result.moved, result.failed), (0, 1))
        self.assertEqual(move.src.read_text(encoding="utf-8"), "important original content")
        self.assertEqual(move.dst.read_text(encoding="utf-8"), "impor")
        with self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (0, 1))
        self.assertTrue(move.src.exists())
        self.assertTrue(move.dst.exists())

    def test_source_modified_during_copy_is_not_deleted(self):
        move = self.make_move()
        copyfileobj = fs.shutil.copyfileobj

        def change_source(reader, writer, length):
            copyfileobj(reader, writer, length)
            move.src.write_text("updated source must not be deleted", encoding="utf-8")

        with self.cross_device_patch(), mock.patch.object(
            fs.shutil, "copyfileobj", side_effect=change_source
        ):
            with self.assertLogs("smart-sort", level="ERROR"):
                result = fs.apply_moves(self.root, [move], {})

        self.assertEqual((result.moved, result.failed), (0, 1))
        self.assertEqual(move.src.read_text(encoding="utf-8"), "updated source must not be deleted")
        self.assertEqual(move.dst.read_text(encoding="utf-8"), "important original content")

    def test_interrupted_undo_completion_is_idempotently_recovered(self):
        move = self.make_move()
        result = fs.apply_moves(self.root, [move], {})
        append = fs._append_record

        def fail_undo_completion(handle, record):
            if record["type"] == "undo":
                raise OSError(errno.ENOSPC, "disk full")
            return append(handle, record)

        with mock.patch.object(fs, "_append_record", side_effect=fail_undo_completion):
            with self.assertLogs("smart-sort", level="ERROR"):
                self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (1, 1))

        self.assertTrue(move.src.exists())
        self.assertFalse(move.dst.exists())
        self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (1, 0))
        self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (0, 0))

    def test_failed_undo_intent_keeps_the_sorted_file(self):
        move = self.make_move()
        result = fs.apply_moves(self.root, [move], {})
        append = fs._append_record

        def fail_undo_intent(handle, record):
            if record["type"] == "undo_intent":
                raise OSError(errno.ENOSPC, "disk full")
            return append(handle, record)

        with mock.patch.object(fs, "_append_record", side_effect=fail_undo_intent):
            with self.assertLogs("smart-sort", level="ERROR"):
                self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (0, 1))
        self.assertTrue(move.dst.exists())
        self.assertFalse(move.src.exists())
        self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (1, 0))

    def test_undo_keeps_replacement_and_restores_under_new_name(self):
        move = self.make_move()
        result = fs.apply_moves(self.root, [move], {})
        move.src.write_text("replacement", encoding="utf-8")

        self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (1, 0))
        self.assertEqual(move.src.read_text(encoding="utf-8"), "replacement")
        restored = self.root / "document (restored).txt"
        self.assertEqual(restored.read_text(encoding="utf-8"), "important original content")
        self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (0, 0))

    def test_destination_created_during_undo_is_not_overwritten(self):
        move = self.make_move()
        result = fs.apply_moves(self.root, [move], {})
        primitive = "rename" if os.name == "nt" else "link"
        actual = getattr(fs.os, primitive)

        def race(source, destination, *args, **kwargs):
            Path(destination).write_text("new original file", encoding="utf-8")
            return actual(source, destination, *args, **kwargs)

        with mock.patch.object(fs.os, primitive, side_effect=race):
            with self.assertLogs("smart-sort", level="ERROR"):
                self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (0, 1))
        self.assertEqual(move.src.read_text(encoding="utf-8"), "new original file")
        self.assertEqual(move.dst.read_text(encoding="utf-8"), "important original content")
        with self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (0, 1))

    def test_changed_sorted_file_is_not_restored(self):
        move = self.make_move()
        result = fs.apply_moves(self.root, [move], {})
        move.dst.write_text("changed since sorting", encoding="utf-8")

        with self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (0, 1))
        self.assertFalse(move.src.exists())
        self.assertEqual(move.dst.read_text(encoding="utf-8"), "changed since sorting")

    def test_truncated_completion_is_repaired_and_recovered(self):
        move = self.make_move()
        result = fs.apply_moves(self.root, [move], {})
        lines = result.manifest.read_bytes().splitlines(keepends=True)
        result.manifest.write_bytes(b"".join(lines[:2]) + b'{"type": "move"')

        with self.assertLogs("smart-sort", level="WARNING"):
            self.assertEqual(fs.undo_manifest(result.manifest, dry_run=False), (1, 0))
        self.assertTrue(move.src.exists())
        self.assertEqual(
            [record["type"] for record in self.records(result.manifest)],
            ["header", "intent", "undo_intent", "undo"],
        )

    def test_legacy_manifest_can_be_undone(self):
        move = self.make_move()
        move.dst.parent.mkdir()
        os.rename(move.src, move.dst)
        manifest = self.root / "legacy.jsonl"
        entries = [
            {"type": "header", "version": 1, "root": str(self.root)},
            {
                "type": "move",
                "src": move.src.relative_to(self.root).as_posix(),
                "dst": move.dst.relative_to(self.root).as_posix(),
                "size": move.dst.stat().st_size,
                "sha256": fs.sha256_file(move.dst),
            },
        ]
        manifest.write_text("".join(json.dumps(record) + "\n" for record in entries),
                            encoding="utf-8")

        self.assertEqual(fs.undo_manifest(manifest, dry_run=False), (1, 0))
        self.assertTrue(move.src.exists())
        self.assertEqual(fs.undo_manifest(manifest, dry_run=False), (0, 0))

    def test_manifest_paths_cannot_escape_the_root(self):
        move = self.make_move()
        move.dst.parent.mkdir()
        os.rename(move.src, move.dst)
        manifest = self.root / "unsafe.jsonl"
        entries = [
            {"type": "header", "version": 1, "root": str(self.root)},
            {
                "type": "move",
                "src": "../outside.txt",
                "dst": move.dst.relative_to(self.root).as_posix(),
                "size": move.dst.stat().st_size,
                "sha256": fs.sha256_file(move.dst),
            },
        ]
        manifest.write_text("".join(json.dumps(record) + "\n" for record in entries),
                            encoding="utf-8")

        with self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(fs.undo_manifest(manifest, dry_run=False), (0, 1))
        self.assertTrue(move.dst.exists())

    def test_active_lock_rejects_another_process_and_is_released_after_exit(self):
        move = self.make_move()
        code = (
            "import sys; from pathlib import Path; import file_safety as f; "
            "root = f.resolve_root(Path(sys.argv[1])); "
            "m = f.Move(root / 'document.txt', root / 'Cluster_1' / 'document.txt', 0); "
            "f.apply_moves(root, [m], {})"
        )
        with fs.root_lock(fs.resolve_root(self.root)):
            child = subprocess.run(
                [sys.executable, "-c", code, str(self.root)],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("Another Smart-Sort operation is active", child.stderr)
        self.assertTrue(move.src.exists())
        self.assertEqual(fs.apply_moves(self.root, [move], {}).moved, 1)

    def test_crashed_lock_owner_does_not_leave_a_stale_lock(self):
        code = (
            "import os, sys; from pathlib import Path; import file_safety as f; "
            "lock = f.root_lock(f.resolve_root(Path(sys.argv[1]))); "
            "lock.__enter__(); os._exit(0)"
        )
        child = subprocess.run([sys.executable, "-c", code, str(self.root)], check=False)
        self.assertEqual(child.returncode, 0)
        with fs.root_lock(fs.resolve_root(self.root)):
            pass

    def test_rapid_runs_have_distinct_manifests(self):
        manifests = []
        for index in range(3):
            move = self.make_move(name=f"document-{index}.txt")
            result = fs.apply_moves(self.root, [move], {})
            self.assertEqual((result.moved, result.failed), (1, 0))
            manifests.append(result.manifest)
        self.assertEqual(len(set(manifests)), 3)
        self.assertTrue(all(path.exists() for path in manifests))

    def test_discovery_and_destination_reject_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as external:
            target = Path(external)
            (target / "outside.txt").write_text("outside", encoding="utf-8")
            link = self.root / "linked"
            try:
                os.symlink(target, link, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"Creating a directory symlink is unavailable: {exc}")
            self.addCleanup(lambda: link.is_symlink() and link.unlink())
            self.assertEqual(ss.collect_files(self.root, recursive=True), [])
            (target / "nested").mkdir()
            with self.assertRaises(ValueError):
                fs.resolve_root(link / "nested")
            with self.assertRaises(ValueError):
                ss.collect_files(link / "nested", recursive=True)

            link.unlink()
            cluster = self.root / "Cluster_1"
            os.symlink(target, cluster, target_is_directory=True)
            self.addCleanup(lambda: cluster.is_symlink() and cluster.unlink())
            move = self.make_move()
            with self.assertLogs("smart-sort", level="ERROR"):
                result = fs.apply_moves(self.root, [move], {})
            self.assertEqual((result.moved, result.failed), (0, 1))
            self.assertTrue(move.src.exists())
            self.assertFalse((target / move.src.name).exists())

            state_root = self.root / "state-root"
            state_root.mkdir()
            state_link = state_root / fs.STATE_DIR
            os.symlink(target, state_link, target_is_directory=True)
            self.addCleanup(lambda: state_link.is_symlink() and state_link.unlink())
            state_source = state_root / "file.txt"
            state_source.write_text("unchanged", encoding="utf-8")
            with self.assertRaises(ValueError):
                fs.apply_moves(
                    state_root,
                    [fs.Move(state_source, state_root / "Cluster_1" / "file.txt", 0)],
                    {},
                )
            with self.assertRaises(ValueError):
                ss.EmbeddingCache(state_link / "embeddings.npz", {})
            self.assertTrue(state_source.exists())

    @unittest.skipUnless(os.name == "nt", "Windows junctions")
    def test_discovery_and_destination_reject_windows_junctions(self):
        with tempfile.TemporaryDirectory() as external:
            target = Path(external)
            (target / "outside.txt").write_text("outside", encoding="utf-8")
            linked = self.root / "linked"
            cluster = self.root / "Cluster_1"

            def junction(path):
                result = subprocess.run(
                    [os.environ["COMSPEC"], "/c", "mklink", "/J", str(path), str(target)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode:
                    self.skipTest(f"Creating a junction is unavailable: {result.stderr}")

            junction(linked)
            try:
                self.assertEqual(ss.collect_files(self.root, recursive=True), [])
                (target / "nested").mkdir()
                with self.assertRaises(ValueError):
                    fs.resolve_root(linked / "nested")
                with self.assertRaises(ValueError):
                    ss.collect_files(linked / "nested", recursive=True)
                junction(cluster)
                try:
                    move = self.make_move()
                    with self.assertLogs("smart-sort", level="ERROR"):
                        result = fs.apply_moves(self.root, [move], {})
                    self.assertEqual((result.moved, result.failed), (0, 1))
                    self.assertTrue(move.src.exists())
                    self.assertFalse((target / move.src.name).exists())
                finally:
                    if cluster.exists():
                        os.rmdir(cluster)

                state_root = self.root / "state-root"
                state_root.mkdir()
                state_link = state_root / fs.STATE_DIR
                junction(state_link)
                try:
                    state_source = state_root / "file.txt"
                    state_source.write_text("unchanged", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        fs.apply_moves(
                            state_root,
                            [fs.Move(state_source, state_root / "Cluster_1" / "file.txt", 0)],
                            {},
                        )
                    with self.assertRaises(ValueError):
                        ss.EmbeddingCache(state_link / "embeddings.npz", {})
                    self.assertTrue(state_source.exists())
                finally:
                    if state_link.exists():
                        os.rmdir(state_link)
            finally:
                if linked.exists():
                    os.rmdir(linked)

    def test_printed_manifest_path_works_outside_selected_folder(self):
        source_root = self.root / "folder $test's stuff"
        source_root.mkdir()
        for name in ("one.txt", "two.txt"):
            (source_root / name).write_text(
                "invoice payment and ordinary document content", encoding="utf-8"
            )
        output = io.StringIO()
        with redirect_stdout(output):
            code = ss.main([str(source_root), "--backend", "tfidf", "--k", "1", "--apply"])
        self.assertEqual(code, 0)
        manifest = next(
            line.removeprefix("Manifest: ")
            for line in output.getvalue().splitlines()
            if line.startswith("Manifest: ")
        )
        self.assertTrue(Path(manifest).is_absolute())
        command = next(
            line.removeprefix("Undo or recover with: ")
            for line in output.getvalue().splitlines()
            if line.startswith("Undo or recover with: ")
        )
        shell = (
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
            if os.name == "nt" else ["sh", "-c", command]
        )
        completed = subprocess.run(shell, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((source_root / "one.txt").exists())
        self.assertTrue((source_root / "two.txt").exists())


if __name__ == "__main__":
    unittest.main()

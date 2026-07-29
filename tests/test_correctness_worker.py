"""Focused tests for the bounded correctness-worker protocol."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from wlz_optimizer.correctness_worker import WorkerRequest, run_worker


class CorrectnessWorkerTests(unittest.TestCase):
    def request(self, code: str, **overrides) -> WorkerRequest:
        values = {
            "argv": (sys.executable, "-c", code),
            "timeout_seconds": 2.0,
            "max_output_bytes": 4096,
        }
        values.update(overrides)
        return WorkerRequest(**values)

    def test_nonzero_exit_is_completed_and_temp_cwd_is_removed(self) -> None:
        result = run_worker(
            self.request(
                "import os, pathlib, sys; "
                "pathlib.Path('created.txt').write_text('ok'); "
                "print(os.getcwd()); print('problem', file=sys.stderr); sys.exit(7)"
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.returncode, 7)
        self.assertIn("wlz-correctness-worker-", result.stdout.text)
        self.assertEqual(result.stderr.text.strip(), "problem")
        self.assertFalse(Path(result.working_directory).exists())

    def test_stdout_and_stderr_are_drained_but_bounded(self) -> None:
        size = 200_000
        result = run_worker(
            self.request(
                f"import os; os.write(1, b'a' * {size}); os.write(2, b'b' * {size})",
                max_output_bytes=1024,
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.returncode, 0)
        for output, character in ((result.stdout, "a"), (result.stderr, "b")):
            self.assertEqual(output.text, character * 1024)
            self.assertEqual(output.total_bytes, size)
            self.assertTrue(output.truncated)

    def test_timeout_kills_process_group_and_next_run_is_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "orphan.txt"
            child = (
                "import time, pathlib; time.sleep(0.3); "
                f"pathlib.Path({str(marker)!r}).write_text('orphan')"
            )
            parent = (
                "import subprocess, sys, time; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(10)"
            )
            timed_out = run_worker(
                self.request(parent, timeout_seconds=0.1, max_output_bytes=128)
            )
            healthy = run_worker(self.request("print('healthy')"))
            time.sleep(0.4)

            self.assertEqual(timed_out.status, "timeout")
            self.assertIsNotNone(timed_out.returncode)
            self.assertFalse(marker.exists())
            self.assertFalse(Path(timed_out.working_directory).exists())
            self.assertEqual(healthy.status, "completed")
            self.assertEqual(healthy.stdout.text.strip(), "healthy")

    def test_request_roundtrip_and_invalid_limits(self) -> None:
        request = self.request("print('ok')")
        restored = WorkerRequest.from_dict(json.loads(json.dumps(request.to_dict())))
        self.assertEqual(restored, request)
        with self.assertRaisesRegex(ValueError, "argv must be a list"):
            WorkerRequest.from_dict({**request.to_dict(), "argv": "python"})

        for overrides in (
            {"argv": ()},
            {"timeout_seconds": 0},
            {"timeout_seconds": float("inf")},
            {"max_output_bytes": -1},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.request("pass", **overrides)

    def test_launch_error_is_structured_and_bounded(self) -> None:
        result = run_worker(
            WorkerRequest(
                argv=("/definitely/missing/wlz-worker-command",),
                timeout_seconds=1.0,
                max_output_bytes=16,
            )
        )

        self.assertEqual(result.status, "launch_error")
        self.assertIsNone(result.returncode)
        self.assertEqual(result.stdout.total_bytes, 0)
        self.assertLessEqual(len(result.stderr.text.encode("utf-8")), 16)
        self.assertTrue(result.stderr.truncated)
        self.assertFalse(Path(result.working_directory).exists())


if __name__ == "__main__":
    unittest.main()

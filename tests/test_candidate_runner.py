"""Focused tests for staged, isolated candidate execution."""

import inspect
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from wlz_optimizer.candidate_runner import CandidateRunRequest, run_candidate
from wlz_optimizer.hash_utils import sha256_text


def request(code, **overrides):
    values = {
        "candidate_id": "candidate-1",
        "code": code,
        "code_hash": sha256_text(code),
        "entrypoint": "run",
        "payload": {"value": 4},
        "timeout_seconds": 2.0,
        "max_output_bytes": 2048,
    }
    values.update(overrides)
    return CandidateRunRequest(**values)


class CandidateRunnerTests(unittest.TestCase):
    def test_import_gate_stops_after_import_and_rejects_decorator_set(self):
        imported = run_candidate(request(
            "raise RuntimeError('must not run')\ndef run(payload): return payload",
            stop_after_import=True,
        ))
        decorator_error = run_candidate(request(
            "def decorator(value): return lambda function: function\n"
            "@decorator({{'HAS_VALUE': lambda args: True}})\n"
            "def run(payload): return payload\n",
            stop_after_import=True,
        ))

        self.assertEqual((imported.status, imported.phase), ("import_error", "module_import"))
        self.assertEqual(
            (decorator_error.status, decorator_error.phase),
            ("import_error", "module_import"),
        )
        self.assertEqual(decorator_error.error_type, "TypeError")
        self.assertIn("unhashable type: 'dict'", decorator_error.error_message)

        successful = run_candidate(request(
            "def run(payload): raise RuntimeError('must not run')",
            stop_after_import=True,
        ))
        self.assertEqual((successful.status, successful.phase), ("imported", "module_import"))
        self.assertIsNone(successful.value)

    def test_old_selective_decorator_error_is_rejected_by_import_gate(self):
        python_executable = os.environ.get("WLZ_TRITON_PYTHON")
        if not python_executable or not Path(python_executable).is_file():
            self.skipTest("WLZ_TRITON_PYTHON does not name a Torch/Triton Python")
        candidate_path = (
            Path(__file__).resolve().parents[1]
            / "output/real-agent-candidates/_selective_scan_update_kernel/eae3d41b.py"
        )
        code = candidate_path.read_text(encoding="utf-8")
        with patch("wlz_optimizer.candidate_runner.sys.executable", python_executable):
            results = [
                run_candidate(request(
                    code,
                    candidate_id="eae3d41b",
                    entrypoint="selective_state_update",
                    timeout_seconds=20.0,
                    stop_after_import=True,
                ))
                for _ in range(2)
            ]

        for result in results:
            self.assertEqual((result.status, result.phase), ("import_error", "module_import"))
            self.assertEqual(result.error_type, "TypeError")
            self.assertIn("unhashable type: 'dict'", result.error_message)
            self.assertFalse(Path(result.worker_result.working_directory).exists())

    def test_real_file_import_hook_identity_and_cwd_cleanup(self):
        code = """
from pathlib import Path
compiled = False
def __wlz_compile_hook__(payload):
    global compiled
    compiled = True
    payload['value'] = 999
    print('hook log')
def run(payload):
    source = Path(__file__).read_text()
    return {'compiled': compiled, 'file': Path(__file__).name,
            'has_source': 'def run' in source, 'value': payload['value'] + 1}
"""
        result = run_candidate(request(code))

        self.assertEqual((result.status, result.phase), ("completed", "runtime"))
        self.assertEqual(result.candidate_id, "candidate-1")
        self.assertEqual(result.value, {
            "compiled": True, "file": "candidate.py", "has_source": True, "value": 5
        })
        self.assertIn("hook log", result.worker_result.stderr.text)
        self.assertFalse(Path(result.worker_result.working_directory).exists())

    def test_hash_mismatch_does_not_start_worker_and_request_has_no_paths(self):
        with patch("wlz_optimizer.candidate_runner.run_worker") as worker:
            result = run_candidate(request("def run(payload): return payload", code_hash="0" * 64))
        worker.assert_not_called()
        self.assertEqual((result.status, result.phase), ("hash_mismatch", "preflight"))
        self.assertIsNone(result.worker_result)
        parameters = inspect.signature(CandidateRunRequest).parameters
        self.assertNotIn("path", parameters)
        self.assertNotIn("module_name", parameters)

    def test_request_rejects_invalid_identity_payload_and_protocol_limit(self):
        code = "def run(payload): return payload"
        for overrides in (
            {"candidate_id": ""}, {"entrypoint": "os.system"}, {"entrypoint": "class"},
            {"payload": {"bad": float("nan")}}, {"payload": {"bad": object()}},
            {"payload": {"mixed": 1, 2: 3}}, {"code_hash": "bad"},
            {"max_output_bytes": None}, {"max_output_bytes": 2047},
            {"stop_after_import": 1},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                request(code, **overrides)

    def test_business_errors_have_explicit_phases(self):
        cases = (
            ("python_source_error", "python_source_compile", "def run(: pass"),
            ("import_error", "module_import", "raise LookupError('import')\ndef run(p): return p"),
            ("import_error", "module_import", "raise SystemExit(2)"),
            ("entrypoint_error", "entrypoint_resolution", "other = 1"),
            ("entrypoint_error", "entrypoint_resolution", "run = 3"),
            ("local_compile_hook_error", "local_compile_hook",
             "def __wlz_compile_hook__(p): raise ArithmeticError('hook')\ndef run(p): return p"),
            ("runtime_error", "runtime", "def run(p): raise RuntimeError('run')"),
        )
        for status, phase, code in cases:
            with self.subTest(status=status, code=code):
                result = run_candidate(request(code))
                self.assertEqual((result.status, result.phase), (status, phase))
                self.assertEqual(result.candidate_id, "candidate-1")
                self.assertIsNotNone(result.error_type)

    def test_long_import_and_runtime_errors_keep_business_status(self):
        for status, phase, code in (
            ("import_error", "module_import", "raise RuntimeError('x' * 10000)"),
            ("runtime_error", "runtime", "def run(p): raise RuntimeError('y' * 10000)"),
        ):
            with self.subTest(status=status):
                result = run_candidate(request(code))
                self.assertEqual((result.status, result.phase), (status, phase))
                self.assertLessEqual(len(result.error_message.encode()), 512)
                self.assertFalse(result.worker_result.stdout.truncated)

    def test_forged_fd1_protocol_is_not_accepted(self):
        forged = (
            "import os\n"
            "os.write(1, b'{\"candidate_id\":\"candidate-1\",\"status\":\"completed\","
            "\"phase\":\"runtime\",\"value\":\"forged\"}')\n"
            "os._exit(0)\n"
        )
        result = run_candidate(request(forged))

        self.assertEqual((result.status, result.phase), ("worker_error", "protocol"))
        self.assertNotEqual(result.value, "forged")
        self.assertIn("forged", result.worker_result.stderr.text)

    def test_normal_exit_descendant_is_killed_and_next_candidate_is_healthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "orphan.txt"
            child = (
                "import pathlib, time; time.sleep(0.3); "
                f"pathlib.Path({str(marker)!r}).write_text('orphan')"
            )
            code = (
                "import subprocess, sys\n"
                "def run(p):\n"
                f" subprocess.Popen([sys.executable, '-c', {child!r}])\n"
                " return 'parent-completed'\n"
            )
            started = time.monotonic()
            first = run_candidate(request(code, timeout_seconds=0.2))
            elapsed = time.monotonic() - started
            healthy = run_candidate(request("def run(p): return 'healthy'"))
            time.sleep(0.4)

            self.assertEqual((first.status, first.phase), ("completed", "runtime"))
            self.assertLess(elapsed, 0.5)
            self.assertFalse(marker.exists())
            self.assertEqual(healthy.value, "healthy")

    def test_timeout_logs_and_cleanup_are_preserved(self):
        timed_out = run_candidate(request(
            "import time\ndef run(p): time.sleep(10)", timeout_seconds=0.1
        ))
        logged = run_candidate(request(
            "print('z' * 10000)\ndef run(p): return p", max_output_bytes=2048
        ))

        self.assertEqual((timed_out.status, timed_out.phase), ("timeout", "candidate_process"))
        self.assertEqual((logged.status, logged.phase), ("completed", "runtime"))
        self.assertTrue(logged.worker_result.stderr.truncated)
        self.assertEqual(len(logged.worker_result.stderr.text.encode()), 2048)
        self.assertFalse(Path(timed_out.worker_result.working_directory).exists())

    def test_worker_crash_and_nonfinite_result_are_protocol_errors(self):
        crashed = run_candidate(request("import os\nos._exit(3)"))
        nonfinite = run_candidate(request("def run(p): return float('nan')"))

        self.assertEqual((crashed.status, crashed.phase), ("worker_error", "worker"))
        self.assertEqual(crashed.worker_result.returncode, 3)
        self.assertEqual((nonfinite.status, nonfinite.phase), ("worker_error", "protocol"))
        self.assertIsNone(nonfinite.value)


if __name__ == "__main__":
    unittest.main()

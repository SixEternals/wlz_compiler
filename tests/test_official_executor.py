import importlib.util
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_PATH = ROOT / "work" / "official_triton_agent" / "executor.py"


def load_executor_module():
    spec = importlib.util.spec_from_file_location("official_executor_isolation_test", EXECUTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {EXECUTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    config = types.ModuleType("config")
    config.EAConfig = type("EAConfig", (), {})
    with patch.dict(sys.modules, {"config": config}):
        spec.loader.exec_module(module)
    return module


class OfficialExecutorIsolationTests(unittest.TestCase):
    def test_msprof_runs_are_isolated_and_use_current_interpreter(self) -> None:
        module = load_executor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "test script.py"
            test_file.write_text("pass\n", encoding="utf-8")
            executor = module.TritonExecutor(
                baseline_time=1.0,
                test_code_path=str(test_file),
                config=SimpleNamespace(),
                kernel_name="kernel with spaces",
                work_dir=root,
            )
            calls = []

            def fake_run(command, **kwargs):
                calls.append((command, kwargs))
                if len(calls) == 2:
                    return SimpleNamespace(returncode=0)
                output = Path(next(item for item in command if item.startswith("--output=")).split("=", 1)[1])
                opprof = output / "OPPROF_fake"
                opprof.mkdir()
                (opprof / "OpBasicInfo.csv").write_text(
                    "Op Name,Task Duration(us),\n"
                    "kernel with spaces,1.25,\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with patch.object(module.subprocess, "run", side_effect=fake_run):
                first = executor._run_msprof(test_file)
                second = executor._run_msprof(test_file)

            self.assertEqual((first, second), (1.25, None))
            self.assertEqual(len(calls), 2)
            result_dirs = [
                Path(next(item for item in command if item.startswith("--output=")).split("=", 1)[1])
                for command, _ in calls
            ]
            self.assertNotEqual(result_dirs[0], result_dirs[1])
            self.assertTrue(all(command[0:2] == ["msprof", "op"] for command, _ in calls))
            self.assertTrue(all(kwargs["shell"] is False for _, kwargs in calls))
            self.assertTrue(
                all(
                    item == f"--application={sys.executable} {test_file.resolve()}"
                    for command, _ in calls
                    for item in command
                    if item.startswith("--application=")
                )
            )

    def test_profile_parser_selects_first_exact_kernel_record(self) -> None:
        module = load_executor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = module.TritonExecutor(
                baseline_time=1.0,
                test_code_path=str(root / "test.py"),
                config=SimpleNamespace(),
                kernel_name="target_kernel",
                work_dir=root,
            )
            result_dir = root / "profile"
            opprof = result_dir / "OPPROF_fake"
            opprof.mkdir(parents=True)
            (opprof / "OpBasicInfo.csv").write_text(
                "Op Name,Task Duration(us),\n"
                "helper_kernel,99.0,\n"
                "target_kernel,2.5,\n"
                "target_kernel,7.5,\n",
                encoding="utf-8",
            )

            self.assertEqual(executor._parse_op_basic_info(result_dir), 2.5)
            observation = executor.last_profile_observation
            csv_bytes = (opprof / "OpBasicInfo.csv").read_bytes()
            self.assertEqual(observation["execution_time_us"], 2.5)
            self.assertEqual(observation["target_row_index"], 2)
            self.assertEqual(observation["run_directory_id"], "profile")
            self.assertEqual(
                observation["csv_path"], "profile/OPPROF_fake/OpBasicInfo.csv"
            )
            self.assertEqual(
                observation["csv_sha256"], hashlib.sha256(csv_bytes).hexdigest()
            )
            self.assertFalse(Path(observation["csv_path"]).is_absolute())
            facts = observation["toolchain_fingerprint"]["facts"]
            payload = json.dumps(
                facts, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            self.assertEqual(
                observation["toolchain_fingerprint"]["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )

    def test_profile_parser_fails_closed_for_invalid_target_record(self) -> None:
        module = load_executor_module()
        invalid_values = ("", "text", "0", "-1", "nan", "inf")
        for invalid in invalid_values:
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                executor = module.TritonExecutor(
                    baseline_time=1.0,
                    test_code_path=str(root / "test.py"),
                    config=SimpleNamespace(),
                    kernel_name="target_kernel",
                    work_dir=root,
                )
                result_dir = root / "profile"
                opprof = result_dir / "OPPROF_fake"
                opprof.mkdir(parents=True)
                (opprof / "OpBasicInfo.csv").write_text(
                    "Op Name,Task Duration(us)\n"
                    f"target_kernel,{invalid}\n"
                    "target_kernel,1.0\n",
                    encoding="utf-8",
                )

                self.assertIsNone(executor._parse_op_basic_info(result_dir))

    def test_profile_parser_rejects_oversize_escape_and_ambiguous_output(self) -> None:
        module = load_executor_module()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            executor = module.TritonExecutor(
                baseline_time=1.0,
                test_code_path=str(root / "test.py"),
                config=SimpleNamespace(),
                kernel_name="target_kernel",
                work_dir=root,
            )

            oversize = root / "oversize" / "OPPROF_fake"
            oversize.mkdir(parents=True)
            (oversize / "OpBasicInfo.csv").write_bytes(
                b"Op Name,Task Duration(us)\n" + b"x" * (module.MAX_PROFILE_CSV_BYTES + 1)
            )
            self.assertIsNone(executor._parse_op_basic_info(oversize.parent))

            escaped = root / "escaped" / "OPPROF_fake"
            escaped.mkdir(parents=True)
            outside_csv = Path(outside) / "OpBasicInfo.csv"
            outside_csv.write_text(
                "Op Name,Task Duration(us)\ntarget_kernel,1.0\n", encoding="utf-8"
            )
            (escaped / "OpBasicInfo.csv").symlink_to(outside_csv)
            self.assertIsNone(executor._parse_op_basic_info(escaped.parent))

            ambiguous = root / "ambiguous"
            for name in ("OPPROF_one", "OPPROF_two"):
                directory = ambiguous / name
                directory.mkdir(parents=True)
                (directory / "OpBasicInfo.csv").write_text(
                    "Op Name,Task Duration(us)\ntarget_kernel,1.0\n",
                    encoding="utf-8",
                )
            self.assertIsNone(executor._parse_op_basic_info(ambiguous))

    def test_profile_parser_requires_expected_columns_and_target(self) -> None:
        module = load_executor_module()
        csv_cases = (
            "Task Duration(us)\n1.0\n",
            "Op Name\ntarget_kernel\n",
            "Op Name,Task Duration(us)\nhelper_kernel,1.0\n",
            "Op Name,Op Name,Task Duration(us)\ntarget_kernel,target_kernel,1.0\n",
        )
        for content in csv_cases:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                executor = module.TritonExecutor(
                    baseline_time=1.0,
                    test_code_path=str(root / "test.py"),
                    config=SimpleNamespace(),
                    kernel_name="target_kernel",
                    work_dir=root,
                )
                result_dir = root / "profile"
                opprof = result_dir / "OPPROF_fake"
                opprof.mkdir(parents=True)
                (opprof / "OpBasicInfo.csv").write_text(content, encoding="utf-8")

                self.assertIsNone(executor._parse_op_basic_info(result_dir))


if __name__ == "__main__":
    unittest.main()

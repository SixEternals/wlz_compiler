import hashlib
import io
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.output_contract import validate_output_contract
from wlz_optimizer.budget import BudgetController, BudgetLimits
from wlz_optimizer.official_adapter import (
    bind_official_task_failures,
    parse_official_task_failures,
)
from wlz_optimizer.stdlib_llm import StdlibOpenAIClient


def load_candidate_generator():
    path = ROOT / "scripts" / "generate_official_candidate.py"
    spec = importlib.util.spec_from_file_location("candidate_generator_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_agent_smoke_packager():
    path = ROOT / "scripts" / "build_official_agent_smoke.py"
    spec = importlib.util.spec_from_file_location("agent_smoke_packager_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OutputContractTests(unittest.TestCase):
    def _dataset(self, root: Path, *operators: str) -> Path:
        datasets = root / "datasets"
        for operator in operators:
            operator_dir = datasets / operator
            operator_dir.mkdir(parents=True)
            (operator_dir / f"{operator}.py").write_text("def baseline():\n    pass\n")
        return datasets

    def test_missing_output_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = validate_output_contract(root / "output", self._dataset(root, "op_a"))
            self.assertFalse(report["valid"])
            self.assertEqual(report["errors"][0]["code"], "output_root_missing")

    def test_valid_numeric_layout_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a", "op_b")
            output = root / "output"
            for operator in ("op_a", "op_b"):
                operator_dir = output / operator
                operator_dir.mkdir(parents=True)
                (operator_dir / f"{operator}_1.py").write_text("def candidate():\n    pass\n")
                (operator_dir / f"{operator}_stats.json").write_text(json.dumps({"ok": True}))
            self.assertTrue(validate_output_contract(output, datasets)["valid"])

    def test_missing_operator_and_invalid_candidate_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a", "op_b")
            operator_dir = root / "output" / "op_a"
            operator_dir.mkdir(parents=True)
            (operator_dir / "op_a_1.py").write_text("def broken(:\n")
            (operator_dir / "op_a_stats.json").write_text("not json")
            report = validate_output_contract(root / "output", datasets)
            codes = {item["code"] for item in report["errors"]}
            self.assertEqual(
                codes,
                {"candidate_python_invalid", "operator_dir_missing", "stats_json_invalid"},
            )

    def test_reference_v_requires_explicit_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a")
            operator_dir = root / "output" / "op_a"
            operator_dir.mkdir(parents=True)
            (operator_dir / "op_a_v1.py").write_text("def candidate():\n    pass\n")
            (operator_dir / "op_a_best.py").write_text("def candidate():\n    pass\n")
            (operator_dir / "op_a_stats.json").write_text(json.dumps({"ok": True}))
            self.assertFalse(validate_output_contract(root / "output", datasets)["valid"])
            self.assertTrue(
                validate_output_contract(root / "output", datasets, "reference-v")["valid"]
            )

    def test_reference_v_requires_best_and_stats_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a")
            operator_dir = root / "output" / "op_a"
            operator_dir.mkdir(parents=True)
            (operator_dir / "op_a_v1.py").write_text("def candidate():\n    pass\n")
            report = validate_output_contract(root / "output", datasets, "reference-v")
            self.assertFalse(report["valid"])
            self.assertEqual(
                {item["code"] for item in report["errors"]},
                {"best_candidate_missing", "stats_json_missing"},
            )

    def test_single_kernel_mode_accepts_partial_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a", "op_b")
            operator_dir = root / "output" / "op_a"
            operator_dir.mkdir(parents=True)
            (operator_dir / "op_a_1.py").write_text("def candidate():\n    pass\n")
            report = validate_output_contract(
                root / "output", datasets, kernel="op_a"
            )
            self.assertTrue(report["valid"])
            self.assertEqual(report["expected_operator_count"], 1)

    def test_more_than_five_and_mixed_aliases_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a")
            operator_dir = root / "output" / "op_a"
            operator_dir.mkdir(parents=True)
            for index in range(1, 7):
                (operator_dir / f"op_a_{index}.py").write_text("def candidate():\n    pass\n")
            (operator_dir / "op_a_v1.py").write_text("def duplicate():\n    pass\n")
            report = validate_output_contract(root / "output", datasets)
            codes = {item["code"] for item in report["errors"]}
            self.assertIn("candidate_count_invalid", codes)
            self.assertIn("candidate_alias_conflict", codes)

    def test_packager_writes_traceable_single_candidate_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a")
            source = datasets / "op_a" / "op_a_seed.py"
            source.write_text("def candidate():\n    return 1\n")
            artifact = root / "smoke.zip"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "build_official_output_smoke.py"),
                "--datasets-dir",
                str(datasets),
                "--kernel",
                "op_a",
                "--source",
                str(source),
                "--output-zip",
                str(artifact),
            ]
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            with zipfile.ZipFile(artifact) as archive:
                self.assertEqual(archive.namelist(), ["output/op_a/op_a_1.py"])
            manifest = json.loads(
                artifact.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["source_sha256"], manifest["target_sha256"])
            self.assertEqual(manifest["artifact_kind"], "official-output-format-smoke")
            original_hash = manifest["artifact_sha256"]
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(
                original_hash,
                hashlib.sha256(artifact.read_bytes()).hexdigest(),
            )

    def test_agent_packager_accepts_repository_relative_manifest_path(self) -> None:
        module = load_agent_smoke_packager()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a")
            candidate = root / "candidate.py"
            code = "def candidate():\n    return 1\n"
            candidate.write_text(code, encoding="utf-8")
            manifest = root / "candidate.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "candidate": {
                            "id": "candidate",
                            "op_name": "op_a",
                            "code_hash": hashlib.sha256(code.encode()).hexdigest(),
                            "generation": 1,
                        },
                        "static_evaluation": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            artifact = root / "smoke.zip"
            sidecar = module.build_agent_smoke(
                Path(os.path.relpath(datasets)),
                "op_a",
                Path(os.path.relpath(candidate)),
                Path(os.path.relpath(manifest)),
                Path(os.path.relpath(artifact)),
            )

            self.assertEqual(sidecar["candidate_id"], "candidate")
            self.assertEqual(sidecar["candidate_count"], 1)
            self.assertFalse(Path(sidecar["candidate_manifest_path"]).is_absolute())
            with zipfile.ZipFile(artifact) as archive:
                self.assertIsNone(archive.testzip())
                stats = json.loads(archive.read("output/op_a/op_a_stats.json"))
            summary = stats["top5_summary"][0]
            self.assertEqual(summary["code_hash"], hashlib.sha256(code.encode()).hexdigest())
            self.assertEqual(summary["parent_ids"], [])
            self.assertEqual(
                summary["manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                sidecar["selections"],
                [{
                    "operator": "op_a",
                    "candidate_variant": "op_a_v1",
                    "candidate_id": "candidate",
                    "candidate_sha256": hashlib.sha256(code.encode()).hexdigest(),
                    "candidate_manifest_path": sidecar["candidate_manifest_path"],
                }],
            )
            failures = parse_official_task_failures(
                "=== 失败任务 ===\nop_a tc1 op_a_v1: runtime error (returncode=0)\n"
            )
            bound = bind_official_task_failures(
                failures,
                sidecar,
                sidecar["artifact_sha256"],
                artifact_bytes=artifact.read_bytes(),
            )
            self.assertEqual(bound[0].candidate_id, "candidate")
            self.assertEqual(
                bound[0].candidate_code_hash,
                hashlib.sha256(code.encode()).hexdigest(),
            )

            invalid = json.loads(manifest.read_text(encoding="utf-8"))
            invalid["candidate"]["id"] = ""
            manifest.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty candidate ID"):
                module.build_agent_smoke(
                    datasets,
                    "op_a",
                    candidate,
                    manifest,
                    root / "invalid.zip",
                )

    def test_agent_packager_binds_multiple_candidate_variants(self) -> None:
        module = load_agent_smoke_packager()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a")
            sources = []
            manifests = []
            expected = []
            for index in (1, 2):
                candidate_id = f"candidate-{index}"
                code = f"def candidate():\n    return {index}\n"
                source = root / f"{candidate_id}.py"
                source.write_text(code, encoding="utf-8")
                manifest = root / f"{candidate_id}.manifest.json"
                manifest.write_text(json.dumps({
                    "candidate": {
                        "id": candidate_id,
                        "op_name": "op_a",
                        "code_hash": hashlib.sha256(code.encode()).hexdigest(),
                        "generation": index,
                    },
                    "static_evaluation": {"passed": True},
                }), encoding="utf-8")
                sources.append(source)
                manifests.append(manifest)
                expected.append((candidate_id, code))

            artifact = root / "portfolio.zip"
            sidecar = module.build_agent_smoke(
                datasets, "op_a", sources, manifests, artifact
            )
            with zipfile.ZipFile(artifact) as archive:
                self.assertEqual(
                    archive.read("output/op_a/op_a_best.py"),
                    archive.read("output/op_a/op_a_v1.py"),
                )
                stats = json.loads(archive.read("output/op_a/op_a_stats.json"))
                self.assertEqual(
                    [item["id"] for item in stats["top5_summary"]],
                    [item[0] for item in expected],
                )
            self.assertEqual(sidecar["candidate_count"], 2)
            self.assertEqual(
                [item["candidate_variant"] for item in sidecar["selections"]],
                ["op_a_v1", "op_a_v2"],
            )
            failures = parse_official_task_failures(
                "=== 失败任务 ===\n"
                "op_a tc1 op_a_v1: runtime error (returncode=0)\n"
                "op_a tc2 op_a_v2: accuracy check failed (returncode=0)\n"
            )
            bound = bind_official_task_failures(
                failures,
                sidecar,
                sidecar["artifact_sha256"],
                artifact_bytes=artifact.read_bytes(),
            )
            self.assertEqual(
                [item.candidate_id for item in bound],
                [item[0] for item in expected],
            )
            self.assertEqual(
                [item.candidate_code_hash for item in bound],
                [hashlib.sha256(item[1].encode()).hexdigest() for item in expected],
            )

            with zipfile.ZipFile(artifact) as archive:
                entries = {name: archive.read(name) for name in archive.namelist()}

            def rebuilt(changes):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as archive:
                    for name, data in entries.items():
                        archive.writestr(name, changes.get(name, data))
                return buffer.getvalue()

            best = rebuilt({"output/op_a/op_a_best.py": b"tampered"})
            with self.assertRaisesRegex(ValueError, "best source mismatch"):
                bind_official_task_failures(
                    failures,
                    {**sidecar, "artifact_sha256": hashlib.sha256(best).hexdigest()},
                    hashlib.sha256(best).hexdigest(),
                    artifact_bytes=best,
                )

            stats_path = "output/op_a/op_a_stats.json"
            stats = json.loads(entries[stats_path])
            stats["top5_summary"].reverse()
            swapped = rebuilt({stats_path: json.dumps(stats).encode()})
            with self.assertRaisesRegex(ValueError, "stats identity mismatch"):
                bind_official_task_failures(
                    failures,
                    {**sidecar, "artifact_sha256": hashlib.sha256(swapped).hexdigest()},
                    hashlib.sha256(swapped).hexdigest(),
                    artifact_bytes=swapped,
                )

            gap = json.loads(json.dumps(sidecar))
            gap["selections"][1]["candidate_variant"] = "op_a_v3"
            with self.assertRaisesRegex(ValueError, "contiguous from v1"):
                bind_official_task_failures(
                    failures, gap, sidecar["artifact_sha256"], artifact_bytes=artifact.read_bytes()
                )

            overlay = {
                **sidecar,
                "selections": [],
                "replacements": sidecar["selections"],
                "base_artifact": {"sha256": "a" * 64},
            }
            base = {"artifact_sha256": "a" * 64, "selections": sidecar["selections"]}
            with self.assertRaisesRegex(ValueError, "Multi-version overlay"):
                bind_official_task_failures(
                    failures,
                    overlay,
                    sidecar["artifact_sha256"],
                    artifact_bytes=artifact.read_bytes(),
                    base_manifest=base,
                )

            with self.assertRaisesRegex(ValueError, "between 1 and 5 pairs"):
                module.build_agent_smoke(
                    datasets,
                    "op_a",
                    sources * 3,
                    manifests * 3,
                    root / "too-many.zip",
                )

    def test_agent_packager_requires_complete_b2_admission_and_usage(self) -> None:
        module = load_agent_smoke_packager()
        operator = "_quantize_k_cache_fast_kernel"

        def manifest_for(code):
            return {
                "candidate": {
                    "id": "candidate",
                    "op_name": operator,
                    "code_hash": hashlib.sha256(code.encode()).hexdigest(),
                    "status": "static_pass",
                    "generation": 1,
                    "model_used": "deepseek-v4-pro",
                    "prompt_id": "a" * 64,
                },
                "static_evaluation": {"passed": True},
                "import_evaluation": {
                    "status": "imported",
                    "phase": "module_import",
                },
                "correctness_evaluation": {
                    "admission_policy_id": "local-quantize-k-cache-public-cuda-v1",
                    "status": "passed",
                    "eligible_for_performance": True,
                    "blocking_reasons": [],
                    "evidence_scope": "local_cuda_proxy_only_not_ascend_or_official",
                    "results": [{"returncode": 0, "matrix_completed": True}],
                },
                "llm_stats": {
                    "call_count": 1,
                    "calls": [
                        {
                            "status": "succeeded",
                            "model": "deepseek-v4-pro",
                            "prompt_sha256": "a" * 64,
                            "usage": {"total_tokens": 17},
                        }
                    ],
                },
                "rejection_error": None,
            }

        mutations = {
            "missing_import": lambda value: value.pop("import_evaluation"),
            "wrong_policy": lambda value: value["correctness_evaluation"].update(
                admission_policy_id="other"
            ),
            "failed_correctness": lambda value: value[
                "correctness_evaluation"
            ].update(status="failed", eligible_for_performance=False),
            "missing_marker": lambda value: value["correctness_evaluation"][
                "results"
            ][0].update(matrix_completed=False),
            "zero_usage": lambda value: value["llm_stats"]["calls"][0][
                "usage"
            ].update(total_tokens=0),
            "prompt_mismatch": lambda value: value["llm_stats"]["calls"][0].update(
                prompt_sha256="b" * 64
            ),
        }
        for label in ("valid", *mutations):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                datasets = self._dataset(root, operator)
                code = "def candidate():\n    return 1\n"
                candidate = root / "candidate.py"
                candidate.write_text(code, encoding="utf-8")
                manifest_data = manifest_for(code)
                if label != "valid":
                    mutations[label](manifest_data)
                manifest = root / "candidate.manifest.json"
                manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
                artifact = root / "b2.zip"
                if label == "valid":
                    sidecar = module.build_agent_smoke(
                        datasets, operator, candidate, manifest, artifact
                    )
                    self.assertEqual(sidecar["candidate_id"], "candidate")
                else:
                    with self.assertRaisesRegex(ValueError, "B2 candidate lacks"):
                        module.build_agent_smoke(
                            datasets, operator, candidate, manifest, artifact
                        )
                    self.assertFalse(artifact.exists())

    def test_stdlib_llm_parses_openai_compatible_response(self) -> None:
        budget = BudgetController(BudgetLimits(10_000, 300))
        config = SimpleNamespace(
            llm_models=["deepseek-v4-pro"],
            api_url="https://example.invalid",
            api_key="bounded-test-key",
            llm_temperature=0.2,
            max_llm_tokens=128,
            budget_controller=budget,
            llm_expected_seconds=120,
        )
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "def candidate():\n    pass\n"}}],
                "usage": {"total_tokens": 17},
                "id": "response-secret-id",
                "model": "deepseek-v4-pro-actual",
            }
        ).encode("utf-8")
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            client = StdlibOpenAIClient(config)
            result = client.generate("mutate this", purpose="mutate")
        self.assertIn("def candidate", result)
        self.assertEqual(client.get_stats()["calls"][0]["usage"]["total_tokens"], 17)
        self.assertEqual(budget.snapshot().used_tokens, 17)
        self.assertEqual(budget.snapshot().in_flight_calls, 0)
        record = client.get_stats()["calls"][0]
        self.assertEqual(record["status"], "succeeded")
        self.assertEqual(record["response_model"], "deepseek-v4-pro-actual")
        self.assertEqual(record["retry_count"], 0)
        self.assertEqual(record["timeout_seconds"], 120)
        self.assertGreaterEqual(record["latency_seconds"], 0)
        self.assertEqual(
            record["response_sha256"],
            hashlib.sha256(result.encode("utf-8")).hexdigest(),
        )
        self.assertNotIn("response-secret-id", str(record))
        self.assertNotIn(config.api_url, str(record))
        self.assertNotIn(config.api_key, str(record))
        sent_payload = json.loads(urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(sent_payload["thinking"], {"type": "disabled"})

    def test_stdlib_llm_budget_denial_and_unknown_call_fail_closed(self) -> None:
        denied_budget = BudgetController(BudgetLimits(10, 300))
        config = SimpleNamespace(
            llm_models=["deepseek-v4-pro"],
            api_url="https://example.invalid",
            api_key="bounded-test-key",
            llm_temperature=0.2,
            max_llm_tokens=128,
            budget_controller=denied_budget,
            llm_expected_seconds=120,
        )
        with patch("urllib.request.urlopen") as urlopen:
            with self.assertRaisesRegex(RuntimeError, "budget denied request"):
                StdlibOpenAIClient(config).generate("mutate this")
        urlopen.assert_not_called()
        self.assertEqual(denied_budget.snapshot().used_tokens, 0)

        uncertain_budget = BudgetController(BudgetLimits(10_000, 300))
        config.budget_controller = uncertain_budget
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaisesRegex(RuntimeError, "LLM request failed"):
                StdlibOpenAIClient(config).generate("mutate this")
        snapshot = uncertain_budget.snapshot()
        self.assertGreater(snapshot.used_tokens, config.max_llm_tokens)
        self.assertEqual(snapshot.in_flight_calls, 0)
        self.assertEqual(snapshot.stop_reason, "unknown_in_flight_call")
        config.budget_controller = None
        failure = StdlibOpenAIClient(config)
        with patch("urllib.request.urlopen", side_effect=TimeoutError("secret body")):
            with self.assertRaises(RuntimeError):
                failure.generate("secret prompt", unknown_secret="must-not-record")
        failure_record = failure.call_history[0]
        self.assertEqual(failure_record["status"], "failed")
        self.assertEqual(failure_record["error_type"], "TimeoutError")
        self.assertNotIn("secret body", str(failure_record))
        self.assertNotIn("secret prompt", str(failure_record))
        self.assertNotIn("must-not-record", str(failure_record))

    def test_stdlib_request_fingerprint_covers_non_prompt_request_config(self) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "def candidate():\n    pass\n"}}],
            "usage": {"total_tokens": 17},
        }).encode("utf-8")
        response.__enter__.return_value = response

        fingerprints = []
        prompt_hashes = []
        variants = (
            ("https://provider-a.invalid/v1", "model-a", 0.2, 128, "system-v1"),
            ("https://provider-a.invalid/v1", "model-b", 0.2, 128, "system-v1"),
            ("https://provider-a.invalid/v1", "model-a", 0.3, 128, "system-v1"),
            ("https://provider-a.invalid/v1", "model-a", 0.2, 256, "system-v1"),
            ("https://provider-b.invalid/v1", "model-a", 0.2, 128, "system-v1"),
            ("https://provider-a.invalid/v1", "model-a", 0.2, 128, "system-v2"),
        )
        with patch("urllib.request.urlopen", return_value=response):
            for api_url, model, temperature, max_tokens, version in variants:
                client = StdlibOpenAIClient(SimpleNamespace(
                    llm_models=[model],
                    api_url=api_url,
                    api_key="fingerprint-test-key",
                    llm_temperature=temperature,
                    max_llm_tokens=max_tokens,
                ))
                client.generate(
                    "same user",
                    system_msg="same system",
                    system_prompt_version=version,
                )
                record = client.call_history[0]
                fingerprints.append(record["request_fingerprint"])
                prompt_hashes.append(record["prompt_sha256"])
                self.assertNotIn(api_url, str(record))

        self.assertEqual(len(set(fingerprints)), len(variants))
        self.assertEqual(len(set(prompt_hashes)), 1)

    def test_structure_hash_ignores_diagnostic_text_only(self) -> None:
        module = load_candidate_generator()
        before = '''"""module docs"""
def candidate(x):
    """first docs"""
    # first comment
    assert x, "first assertion"
    return x
'''
        after = '''"""changed module docs"""
def candidate(x):
    """changed docs"""
    # changed comment
    assert x, "changed assertion"
    return x
'''
        self.assertEqual(module._structure_hash(before), module._structure_hash(after))

    def test_structure_hash_preserves_executable_and_launch_literals(self) -> None:
        module = load_candidate_generator()
        base = "def candidate():\n    return 1\n"
        self.assertNotEqual(
            module._structure_hash(base),
            module._structure_hash("def candidate():\n    return 2\n"),
        )
        launch = "def wrapper(kernel, x):\n    return kernel[(1,)](x, num_warps=4, device='npu')\n"
        self.assertNotEqual(
            module._structure_hash(launch),
            module._structure_hash(
                "def wrapper(kernel, x):\n    return kernel[(1,)](x, num_warps=8, device='npu')\n"
            ),
        )
        self.assertNotEqual(
            module._structure_hash(launch),
            module._structure_hash(
                "def wrapper(kernel, x):\n    return kernel[(1,)](x, num_warps=4, device='cuda')\n"
            ),
        )

    def test_structure_hash_ignores_consistent_local_variable_rename(self) -> None:
        module = load_candidate_generator()
        before = "def candidate(x):\n    pid_m = source()\n    return pid_m + x\n"
        renamed = "def candidate(x):\n    program_m = source()\n    return program_m + x\n"
        changed = "def candidate(x):\n    pid_m = source()\n    return x - pid_m\n"

        self.assertEqual(module._structure_hash(before), module._structure_hash(renamed))
        self.assertNotEqual(module._structure_hash(before), module._structure_hash(changed))

    def test_structure_hash_rejects_real_diagnostic_only_candidate(self) -> None:
        module = load_candidate_generator()
        baseline = (
            ROOT
            / "work/official_triton_agent/datasets/_set_k_and_s_triton_kernel"
            / "_set_k_and_s_triton_kernel.py"
        )
        candidate = (
            ROOT
            / "output/real-agent-candidates/_set_k_and_s_triton_kernel/fd113ce1.py"
        )
        if not candidate.is_file():
            self.skipTest("historical generated candidate is not available")
        self.assertEqual(
            module._structure_hash(baseline.read_text(encoding="utf-8")),
            module._structure_hash(candidate.read_text(encoding="utf-8")),
        )
        self.assertNotEqual(
            module._structure_hash("raise ValueError('detail')"),
            module._structure_hash("raise TypeError('detail')"),
        )

    def test_rejected_real_candidate_is_preserved(self) -> None:
        module = load_candidate_generator()

        class FakeConfig:
            def __init__(self):
                self.api_url = None
                self.api_key = None
                self.llm_models = ["fake-model"]

        class FakeLlm:
            def __init__(self, config):
                self.call_history = [{"prompt_sha256": "a" * 64}]

            def get_stats(self):
                return {"call_count": 1}

        class FakeIndividual:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeOperators:
            def __init__(self, llm, config):
                pass

            def mutate(self, parent):
                return SimpleNamespace(
                    code="import triton\nimport triton.language as tl\ndef op_a(x, extra):\n    pass\n",
                    id="rejected-1",
                    generation=1,
                    metadata={"mutation_type": "param_tuning"},
                    model_used="fake-model",
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a")
            (datasets / "op_a" / "test_op_a_1.py").write_text("from op_a import baseline\n")
            output = root / "generated"
            relative_output = Path(os.path.relpath(output, Path.cwd()))
            fake_modules = (
                SimpleNamespace(EAConfig=FakeConfig),
                SimpleNamespace(GeneticOperators=FakeOperators, Individual=FakeIndividual),
                SimpleNamespace(interface_contract_error=lambda baseline, candidate: "signature differs"),
            )
            with patch.object(module, "ROOT", root), patch.object(
                module, "_load_official_modules", return_value=fake_modules
            ), patch.object(module, "StdlibOpenAIClient", FakeLlm):
                with self.assertRaisesRegex(ValueError, "rejected candidate saved"):
                    module.generate_candidate(
                        root,
                        datasets,
                        "op_a",
                        datasets / "op_a" / "op_a.py",
                        relative_output,
                        1,
                    )
            manifest = json.loads(
                (output / "op_a" / "rejected-1.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["candidate"]["status"], "rejected")
            self.assertEqual(manifest["rejection_error"], "Generated candidate failed full interface contract: signature differs")

    def test_parent_identical_candidate_stops_before_static_and_import(self) -> None:
        module = load_candidate_generator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a")
            output = root / "generated"
            parent_path = output / "op_a/parent.py"
            parent_path.parent.mkdir(parents=True)
            parent_code = "def baseline():\n    return 1\n"
            parent_path.write_text(parent_code, encoding="utf-8")
            parent_hash = hashlib.sha256(parent_code.encode()).hexdigest()
            parent_path.with_suffix(".manifest.json").write_text(
                json.dumps({"candidate": {
                    "id": "parent",
                    "op_name": "op_a",
                    "code_hash": parent_hash,
                    "generation": 2,
                    "model_used": "parent-model",
                }}),
                encoding="utf-8",
            )
            contract_check = MagicMock(return_value=None)
            fake_modules = (
                SimpleNamespace(EAConfig=lambda: SimpleNamespace(llm_models=["fake-model"])),
                SimpleNamespace(
                    Individual=SimpleNamespace,
                    GeneticOperators=lambda *_: SimpleNamespace(
                        mutate=lambda parent: SimpleNamespace(
                            code=parent.code,
                            id="no-op-child",
                            generation=1,
                            metadata={"mutation_type": "local_rewrite"},
                            model_used="fake-model",
                        )
                    ),
                ),
                SimpleNamespace(interface_contract_error=contract_check),
            )
            fake_llm = SimpleNamespace(
                call_history=[{"prompt_sha256": "a" * 64}],
                get_stats=lambda: {"call_count": 1},
            )
            with patch.object(
                module, "_load_official_modules", return_value=fake_modules
            ), patch.object(
                module, "StdlibOpenAIClient", return_value=fake_llm
            ), patch.object(
                module.LocalExecutor, "evaluate"
            ) as static_gate, patch.object(module, "run_candidate") as import_gate:
                with self.assertRaisesRegex(ValueError, "byte-identical to its parent"):
                    module.generate_candidate(
                        root, datasets, "op_a", parent_path, output, 1
                    )

            manifest = json.loads(
                (output / "op_a/no-op-child.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                (output / "op_a/no-op-child.py").read_bytes(), parent_path.read_bytes()
            )
            self.assertEqual(manifest["candidate"]["status"], "rejected")
            self.assertEqual(manifest["candidate"]["parent_ids"], ["parent"])
            self.assertEqual(manifest["candidate"]["code_hash"], parent_hash)
            self.assertEqual(manifest["parent_sha256"], parent_hash)
            self.assertEqual(
                manifest["rejection_error"],
                "Generated candidate is byte-identical to its parent",
            )
            self.assertIsNone(manifest["static_evaluation"])
            self.assertIsNone(manifest["import_evaluation"])
            contract_check.assert_not_called()
            static_gate.assert_not_called()
            import_gate.assert_not_called()

    def test_existing_duplicate_candidate_stops_before_static_and_import(self) -> None:
        module = load_candidate_generator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root, "op_a")
            output = root / "generated"
            candidate_dir = output / "op_a"
            candidate_dir.mkdir(parents=True)
            duplicate_code = "def baseline():\n    return 2\n"
            (candidate_dir / "existing.py").write_text(
                duplicate_code, encoding="utf-8"
            )
            contract_check = MagicMock(return_value=None)
            fake_modules = (
                SimpleNamespace(
                    EAConfig=lambda: SimpleNamespace(llm_models=["fake-model"])
                ),
                SimpleNamespace(
                    Individual=SimpleNamespace,
                    GeneticOperators=lambda *_: SimpleNamespace(
                        mutate=lambda parent: SimpleNamespace(
                            code=duplicate_code,
                            id="duplicate-child",
                            generation=1,
                            metadata={"mutation_type": "param_tuning"},
                            model_used="fake-model",
                        )
                    ),
                ),
                SimpleNamespace(interface_contract_error=contract_check),
            )
            fake_llm = SimpleNamespace(
                call_history=[{"prompt_sha256": "a" * 64}],
                get_stats=lambda: {"call_count": 1},
            )
            with patch.object(
                module, "_load_official_modules", return_value=fake_modules
            ), patch.object(
                module, "StdlibOpenAIClient", return_value=fake_llm
            ), patch.object(
                module.LocalExecutor, "evaluate"
            ) as static_gate, patch.object(module, "run_candidate") as import_gate:
                with self.assertRaisesRegex(
                    ValueError, "duplicates existing candidate: existing"
                ):
                    module.generate_candidate(
                        root,
                        datasets,
                        "op_a",
                        datasets / "op_a" / "op_a.py",
                        output,
                        1,
                    )

            manifest = json.loads(
                (candidate_dir / "duplicate-child.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["candidate"]["status"], "rejected")
            self.assertEqual(
                manifest["rejection_error"],
                "Generated candidate duplicates existing candidate: existing",
            )
            contract_check.assert_not_called()
            static_gate.assert_not_called()
            import_gate.assert_not_called()


if __name__ == "__main__":
    unittest.main()

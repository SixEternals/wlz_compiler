import importlib.util
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from wlz_optimizer.hash_utils import sha256_text

ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


def load_module():
    path = ROOT / "scripts" / "build_official_agent_batch_smoke.py"
    spec = importlib.util.spec_from_file_location("batch_smoke_packager_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_generator_module():
    path = ROOT / "scripts" / "generate_official_candidate.py"
    spec = importlib.util.spec_from_file_location("candidate_generator_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchSmokePackagerTests(unittest.TestCase):
    def _write_operator(
        self,
        root: Path,
        operator: str,
        correctness_evaluation=_MISSING,
        candidate_id: str = "candidate",
    ) -> Path:
        datasets = root / "datasets" / operator
        datasets.mkdir(parents=True, exist_ok=True)
        (datasets / f"{operator}.py").write_text(
            f"def {operator}(x):\n    return x\n", encoding="utf-8"
        )
        candidates = root / "candidates" / operator
        candidates.mkdir(parents=True, exist_ok=True)
        code = f"def {operator}(x):\n    return x + 1\n"
        (candidates / f"{candidate_id}.py").write_text(code, encoding="utf-8")
        manifest = {
            "candidate": {
                "id": candidate_id,
                "op_name": operator,
                "code_hash": sha256_text(code),
                "status": "static_pass",
                "generation": 1,
            },
            "static_evaluation": {"passed": True},
            "import_evaluation": {
                "status": "imported",
                "phase": "module_import",
                "error_type": None,
                "error_message": None,
            },
            "rejection_error": None,
            "llm_stats": {"call_count": 1},
            "holdout_evaluation": {
                "status": "passed",
                "split": "holdout",
                "case_signature": "h" * 64,
                "case_count": 1,
                "used_for_search": False,
                "used_in_prompt": False,
            },
        }
        if correctness_evaluation is not _MISSING:
            manifest["correctness_evaluation"] = correctness_evaluation
        manifest_path = candidates / f"{candidate_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def _write_selection_lock(self, root: Path, operators: list[str]) -> Path:
        selections = []
        for operator in operators:
            manifests = sorted((root / "candidates" / operator).glob("*.manifest.json"))
            manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
            candidate = manifest["candidate"]
            selections.append(
                {
                    "operator": operator,
                    "candidate_id": candidate["id"],
                    "candidate_sha256": candidate["code_hash"],
                }
            )
        path = root / "selection.manifest.json"
        path.write_text(json.dumps({"selections": selections}), encoding="utf-8")
        return path

    def _write_variant(
        self,
        root: Path,
        operator: str,
        candidate_id: str,
        expression: str,
        mutation_kind: str | None = None,
    ) -> Path:
        candidate_dir = root / "candidates" / operator
        candidate_dir.mkdir(parents=True, exist_ok=True)
        code = f"def {operator}(x):\n    return {expression}\n"
        candidate_path = candidate_dir / f"{candidate_id}.py"
        candidate_path.write_text(code, encoding="utf-8")
        candidate = {
            "id": candidate_id,
            "op_name": operator,
            "code_hash": sha256_text(code),
            "status": "static_pass",
            "generation": 1,
        }
        if mutation_kind is not None:
            candidate["mutation_kind"] = mutation_kind
        manifest = {
            "candidate": candidate,
            "static_evaluation": {"passed": True},
            "import_evaluation": {
                "status": "imported",
                "phase": "module_import",
                "error_type": None,
                "error_message": None,
            },
            "rejection_error": None,
            "llm_stats": {"call_count": 1},
            "holdout_evaluation": {
                "status": "passed",
                "split": "holdout",
                "case_signature": "h" * 64,
                "case_count": 1,
                "used_for_search": False,
                "used_in_prompt": False,
            },
        }
        manifest_path = candidate_dir / f"{candidate_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def _write_historical_source(self, root: Path, selection: Path) -> Path:
        source_zip = root / "historical.zip"
        with zipfile.ZipFile(source_zip, "w") as archive:
            archive.writestr("output/op_a/op_a_v1.py", "def op_a(x): return x\n")
        document = json.loads(selection.read_text(encoding="utf-8"))
        document.update(
            {
                "artifact_path": source_zip.name,
                "artifact_sha256": hashlib.sha256(source_zip.read_bytes()).hexdigest(),
                "operator_count": len(document["selections"]),
                "candidate_count": len(document["selections"]),
                "scoring_intent": "21-operator-functional-and-performance-smoke",
                "layout": "organizer-save-results-v1",
                "archive_entries": ["output/op_a/op_a_v1.py"],
            }
        )
        path = root / "historical.manifest.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_builds_one_valid_entry_set_per_operator(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op_manifest = self._write_operator(root, "op_a", None)
            self._write_operator(
                root,
                "_per_group_transpose",
                {
                    "status": "passed",
                    "eligible_for_performance": True,
                    "blocking_reasons": [],
                    "decision": {
                        "candidate_id": "candidate",
                        "eligible_for_performance": True,
                        "blocking_reasons": [],
                    },
                },
            )
            artifact = root / "batch.zip"
            selection = self._write_selection_lock(
                root, ["op_a", "_per_group_transpose"]
            )
            sidecar = module.build_batch_smoke(
                root / "datasets", root / "candidates", artifact, selection
            )
            with zipfile.ZipFile(artifact) as archive:
                names = archive.namelist()
                self.assertIsNone(archive.testzip())
                stats = json.loads(archive.read("output/op_a/op_a_stats.json"))
            summary = stats["top5_summary"][0]
            self.assertEqual(summary["code_hash"], sidecar["selections"][1]["candidate_sha256"])
            self.assertEqual(summary["parent_ids"], [])
            self.assertEqual(
                summary["manifest_sha256"],
                hashlib.sha256(op_manifest.read_bytes()).hexdigest(),
            )
            self.assertEqual(sidecar["operator_count"], 2)
            self.assertEqual(sidecar["candidate_count"], 2)
            self.assertEqual(
                sidecar["scoring_intent"],
                "mixed-local-admission-smoke-not-for-official-scoring",
            )
            self.assertFalse(sidecar["official_scoring_ready"])
            self.assertEqual(len(names), 6)
            self.assertIn("output/op_a/op_a_v1.py", names)
            self.assertIn(
                "output/_per_group_transpose/_per_group_transpose_stats.json", names
            )
            admission = {
                item["operator"]: item["admission_level"]
                for item in sidecar["selections"]
            }
            self.assertEqual(
                admission,
                {
                    "_per_group_transpose": "local_correctness_admitted",
                    "op_a": "static_import_only",
                },
            )

    def test_rechecks_interface_contract_before_packaging(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "datasets" / "op_a"
            dataset_dir.mkdir(parents=True)
            baseline = """\
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, n, BLOCK_SIZE: tl.constexpr):
    value = tl.load(x)
    tl.store(out, value)

def op_a(x, out, n):
    grid = (triton.cdiv(n, 128),)
    kernel[grid](x, out, n, BLOCK_SIZE=128, num_warps=4)
    return out
"""
            candidate = baseline.replace(
                "    grid =",
                "    torch.zeros((1,), device=\"npu\")\n    grid =",
            )
            (dataset_dir / "op_a.py").write_text(baseline, encoding="utf-8")
            (dataset_dir / "test_op_a_1.py").write_text(
                "from op_a import op_a\n", encoding="utf-8"
            )
            candidate_dir = root / "candidates" / "op_a"
            candidate_dir.mkdir(parents=True)
            candidate_path = candidate_dir / "candidate.py"
            candidate_path.write_text(candidate, encoding="utf-8")
            manifest = {
                "candidate": {
                    "id": "candidate",
                    "op_name": "op_a",
                    "code_hash": sha256_text(candidate),
                    "status": "static_pass",
                    "generation": 1,
                },
                "static_evaluation": {"passed": True},
                "import_evaluation": {
                    "status": "imported",
                    "phase": "module_import",
                },
                "rejection_error": None,
            }
            manifest_path = candidate_dir / "candidate.manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            selection = root / "selection.manifest.json"
            selection.write_text(
                json.dumps(
                    {
                        "selections": [
                            {
                                "operator": "op_a",
                                "candidate_id": "candidate",
                                "candidate_sha256": sha256_text(candidate),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "interface contract preflight.*wrapper timing surface"
            ):
                module.build_batch_smoke(
                    root / "datasets",
                    root / "candidates",
                    root / "batch.zip",
                    selection,
                )
            self.assertFalse((root / "batch.zip").exists())
            self.assertFalse((root / "batch.manifest.json").exists())

    def test_rejects_oversize_and_unsafe_archives(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_operator(root, "op_a", None)
            selection = self._write_selection_lock(root, ["op_a"])
            artifact = root / "oversize.zip"
            with patch.object(module, "MAX_ARTIFACT_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "platform limit"):
                    module.build_batch_smoke(
                        root / "datasets", root / "candidates", artifact, selection
                    )
            self.assertFalse(artifact.exists())
            self.assertFalse(artifact.with_suffix(".manifest.json").exists())

            unsafe = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe, "w") as archive:
                archive.writestr("../escape.py", "pass\n")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                module._verify_archive(unsafe, ["../escape.py"])

    def test_refuses_operator_without_static_pass_candidate(self) -> None:
        module = load_module()
        for rejection_error in (
            "correctness_failed: failed_case",
            "duplicate admitted code hash",
            "correctness gate error: RuntimeError",
        ):
            with self.subTest(error=rejection_error), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest_path = self._write_operator(
                    root,
                    "_per_group_transpose",
                    {
                        "status": "passed",
                        "eligible_for_performance": True,
                        "blocking_reasons": [],
                    },
                )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["candidate"]["status"] = "rejected"
                manifest["rejection_error"] = rejection_error
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                artifact = root / "batch.zip"
                selection = self._write_selection_lock(
                    root, ["_per_group_transpose"]
                )
                with self.assertRaisesRegex(ValueError, "lacks valid"):
                    module.build_batch_smoke(
                        root / "datasets", root / "candidates", artifact, selection
                    )
                self.assertFalse(artifact.exists())
                self.assertFalse(artifact.with_suffix(".manifest.json").exists())

    def test_gated_operator_requires_consistent_correctness_admission(self) -> None:
        module = load_module()
        cases = (
            ("missing", _MISSING),
            ("null", None),
            (
                "failed",
                {
                    "status": "failed",
                    "eligible_for_performance": False,
                    "blocking_reasons": ["failed_case"],
                },
            ),
            (
                "ineligible",
                {
                    "status": "passed",
                    "eligible_for_performance": False,
                    "blocking_reasons": [],
                },
            ),
            (
                "contradictory",
                {
                    "status": "passed",
                    "eligible_for_performance": True,
                    "blocking_reasons": ["failed_case"],
                },
            ),
            (
                "bad_decision",
                {
                    "status": "passed",
                    "eligible_for_performance": True,
                    "blocking_reasons": [],
                    "decision": {
                        "candidate_id": "other",
                        "eligible_for_performance": False,
                        "blocking_reasons": ["failed_case"],
                    },
                },
            ),
        )
        for label, evaluation in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write_operator(
                    root, "_selective_scan_update_kernel", evaluation
                )
                artifact = root / "batch.zip"
                selection = self._write_selection_lock(
                    root, ["_selective_scan_update_kernel"]
                )
                with self.assertRaisesRegex(ValueError, "lacks valid"):
                    module.build_batch_smoke(
                        root / "datasets", root / "candidates", artifact, selection
                    )
                self.assertFalse(artifact.exists())
                self.assertFalse(artifact.with_suffix(".manifest.json").exists())

    def test_b2_operator_requires_policy_marker_and_trustworthy_usage(self) -> None:
        module = load_module()
        operator = "_quantize_k_cache_fast_kernel"

        def add_b2_evidence(manifest_path):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["candidate"].update(
                model_used="deepseek-v4-pro", prompt_id="a" * 64
            )
            manifest["correctness_evaluation"] = {
                "admission_policy_id": "local-quantize-k-cache-public-cuda-v1",
                "status": "passed",
                "eligible_for_performance": True,
                "blocking_reasons": [],
                "evidence_scope": "local_cuda_proxy_only_not_ascend_or_official",
                "results": [{"returncode": 0, "matrix_completed": True}],
            }
            manifest["llm_stats"] = {
                "call_count": 1,
                "calls": [
                    {
                        "status": "succeeded",
                        "model": "deepseek-v4-pro",
                        "prompt_sha256": "a" * 64,
                        "usage": {"total_tokens": 17},
                    }
                ],
            }
            return manifest

        mutations = {
            "wrong_policy": lambda value: value["correctness_evaluation"].update(
                admission_policy_id="other"
            ),
            "missing_marker": lambda value: value["correctness_evaluation"][
                "results"
            ][0].update(matrix_completed=False),
            "missing_usage": lambda value: value["llm_stats"]["calls"][0].pop(
                "usage"
            ),
            "zero_usage": lambda value: value["llm_stats"]["calls"][0][
                "usage"
            ].update(total_tokens=0),
            "model_mismatch": lambda value: value["llm_stats"]["calls"][0].update(
                model="other"
            ),
        }
        for label in ("valid", *mutations):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest_path = self._write_operator(root, operator)
                manifest = add_b2_evidence(manifest_path)
                if label != "valid":
                    mutations[label](manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                selection = self._write_selection_lock(root, [operator])
                artifact = root / "batch.zip"
                if label == "valid":
                    sidecar = module.build_batch_smoke(
                        root / "datasets",
                        root / "candidates",
                        artifact,
                        selection,
                    )
                    self.assertEqual(sidecar["candidate_count"], 1)
                else:
                    with self.assertRaisesRegex(ValueError, "lacks valid"):
                        module.build_batch_smoke(
                            root / "datasets",
                            root / "candidates",
                            artifact,
                            selection,
                        )
                    self.assertFalse(artifact.exists())

    def test_local_ascend_manual_admission_requires_bound_evidence(self) -> None:
        module = load_module()
        operator = "_count_expert_num_tokens"

        def local_evaluation(artifact_sha256: str) -> dict:
            return {
                "admission_policy_id": module._LOCAL_ASCEND_ADMISSION_POLICY_ID,
                "status": "passed",
                "eligible_for_performance": True,
                "blocking_reasons": [],
                "evidence_scope": "local_ascend_910b4_visible_and_holdout_not_official",
                "results": [
                    {
                        "artifact_path": "correctness.json",
                        "artifact_sha256": artifact_sha256,
                        "split": "search_visible",
                        "status": "passed",
                    }
                ],
                "decision": {
                    "candidate_id": "candidate",
                    "eligible_for_performance": True,
                    "blocking_reasons": [],
                },
            }

        for label in (
            "valid",
            "missing_artifact_sha",
            "tampered_artifact",
            "official_scope",
            "missing_holdout",
            "missing_holdout_split",
            "missing_holdout_used_for_search",
            "missing_holdout_used_in_prompt",
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                evidence_path = root / "correctness.json"
                evidence_path.write_text("passed\n", encoding="utf-8")
                manifest_path = self._write_operator(root, operator)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["correctness_evaluation"] = local_evaluation(
                    hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                )
                if label == "missing_artifact_sha":
                    manifest["correctness_evaluation"]["results"][0].pop("artifact_sha256")
                elif label == "tampered_artifact":
                    evidence_path.write_text("tampered\n", encoding="utf-8")
                elif label == "official_scope":
                    manifest["correctness_evaluation"]["evidence_scope"] = "official_a2_a3"
                elif label == "missing_holdout":
                    manifest.pop("holdout_evaluation")
                elif label == "missing_holdout_split":
                    manifest["holdout_evaluation"].pop("split")
                elif label == "missing_holdout_used_for_search":
                    manifest["holdout_evaluation"].pop("used_for_search")
                elif label == "missing_holdout_used_in_prompt":
                    manifest["holdout_evaluation"].pop("used_in_prompt")
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                selection = self._write_selection_lock(root, [operator])
                artifact = root / "batch.zip"
                if label == "valid":
                    sidecar = module.build_batch_smoke(
                        root / "datasets", root / "candidates", artifact, selection
                    )
                    self.assertEqual(
                        sidecar["selections"][0]["admission_level"],
                        "local_ascend_910b4_manual_evidence",
                    )
                elif label in {"missing_holdout", "missing_holdout_split"}:
                    with self.assertRaisesRegex(
                        ValueError, "holdout admission.*holdout_required"
                    ):
                        module.build_batch_smoke(
                            root / "datasets", root / "candidates", artifact, selection
                        )
                elif label in {
                    "missing_holdout_used_for_search",
                    "missing_holdout_used_in_prompt",
                }:
                    with self.assertRaisesRegex(
                        ValueError, "holdout admission.*holdout_metadata_incomplete"
                    ):
                        module.build_batch_smoke(
                            root / "datasets", root / "candidates", artifact, selection
                        )
                else:
                    with self.assertRaisesRegex(ValueError, "lacks valid"):
                        module.build_batch_smoke(
                            root / "datasets", root / "candidates", artifact, selection
                        )
                self.assertFalse(artifact.exists() and label != "valid")

    def test_refuses_missing_or_failed_import_evidence(self) -> None:
        module = load_module()
        for label, import_evaluation in (
            ("missing", None),
            ("null", None),
            ("static_null", None),
            (
                "failed",
                {
                    "status": "import_error",
                    "phase": "module_import",
                    "error_type": "TypeError",
                    "error_message": "unhashable type: 'dict'",
                },
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write_operator(root, "op_a")
                manifest_path = root / "candidates/op_a/candidate.manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if label == "static_null":
                    manifest["static_evaluation"] = None
                elif label == "missing":
                    manifest.pop("import_evaluation")
                else:
                    manifest["import_evaluation"] = import_evaluation
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                selection = self._write_selection_lock(root, ["op_a"])
                with self.assertRaisesRegex(ValueError, "lacks valid"):
                    module.build_batch_smoke(
                        root / "datasets",
                        root / "candidates",
                        root / "batch.zip",
                        selection,
                    )

    def test_selection_lock_chooses_exact_candidate(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_operator(root, "op_a", candidate_id="aaa")
            selected_manifest = self._write_operator(
                root, "op_a", candidate_id="selected"
            )
            selected = json.loads(selected_manifest.read_text(encoding="utf-8"))["candidate"]
            selection = root / "selection.manifest.json"
            selection.write_text(
                json.dumps(
                    {
                        "selections": [
                            {
                                "operator": "op_a",
                                "candidate_id": selected["id"],
                                "candidate_sha256": selected["code_hash"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            sidecar = module.build_batch_smoke(
                root / "datasets", root / "candidates", root / "batch.zip", selection
            )
            self.assertEqual(sidecar["selections"][0]["candidate_id"], "selected")

    def test_packages_multiple_slots_in_order_without_filling_to_five(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_operator(root, "op_a")
            first = self._write_variant(root, "op_a", "v1", "x + 1", "launch")
            second = self._write_variant(root, "op_a", "v2", "x * 2", "strategy")
            third = self._write_variant(root, "op_a", "v3", "x - 1", "fallback")
            manifests = [json.loads(path.read_text(encoding="utf-8")) for path in (first, second, third)]
            selection = root / "selection.manifest.json"
            selection.write_text(
                json.dumps(
                    {
                        "selections": [
                            {
                                "operator": "op_a",
                                "candidates": [
                                    {
                                        "candidate_id": item["candidate"]["id"],
                                        "candidate_sha256": item["candidate"]["code_hash"],
                                        "modification_type": kind,
                                        "local_ratio": ratio,
                                    }
                                    for item, kind, ratio in zip(
                                        manifests,
                                        ("launch", "strategy", "fallback"),
                                        (0.8, 0.9, 1.0),
                                    )
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            sidecar = module.build_batch_smoke(
                root / "datasets", root / "candidates", root / "batch.zip", selection
            )

            with zipfile.ZipFile(root / "batch.zip") as archive:
                names = set(archive.namelist())
                self.assertEqual(
                    names,
                    {
                        "output/op_a/op_a_best.py",
                        "output/op_a/op_a_stats.json",
                        "output/op_a/op_a_v1.py",
                        "output/op_a/op_a_v2.py",
                        "output/op_a/op_a_v3.py",
                    },
                )
                self.assertEqual(
                    archive.read("output/op_a/op_a_v1.py"),
                    (root / "candidates/op_a/v1.py").read_bytes(),
                )
                stats = json.loads(archive.read("output/op_a/op_a_stats.json"))
            self.assertEqual(sidecar["candidate_count"], 3)
            self.assertEqual(len(sidecar["selections"][0]["candidates"]), 3)
            self.assertEqual(
                [item["candidate_variant"] for item in sidecar["selections"][0]["candidates"]],
                ["op_a_v1", "op_a_v2", "op_a_v3"],
            )
            self.assertEqual(
                [item["modification_type"] for item in stats["top5_summary"]],
                ["launch", "strategy", "fallback"],
            )
            self.assertEqual(
                [item["local_ratio"] for item in sidecar["selections"][0]["candidates"]],
                [0.8, 0.9, 1.0],
            )
            self.assertEqual(
                [item["local_ratio"] for item in stats["top5_summary"]],
                [0.8, 0.9, 1.0],
            )
            self.assertNotIn("output/op_a/op_a_v4.py", names)
            self.assertNotIn("output/op_a/op_a_v5.py", names)

    def test_multi_slot_selection_rejects_baseline_and_unsafe_strategy_pair(self) -> None:
        module = load_module()
        cases = ("baseline", "same_type", "unknown_type")
        for label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write_operator(root, "op_a")
                if label == "baseline":
                    first = self._write_variant(root, "op_a", "baseline", "x")
                    manifests = [json.loads(first.read_text(encoding="utf-8"))]
                else:
                    kind = None if label == "unknown_type" else "launch"
                    first = self._write_variant(root, "op_a", "v1", "x + 1", kind)
                    second = self._write_variant(root, "op_a", "v2", "x * 2", kind)
                    manifests = [
                        json.loads(path.read_text(encoding="utf-8"))
                        for path in (first, second)
                    ]
                selection = root / "selection.manifest.json"
                selection.write_text(
                    json.dumps(
                        {
                            "selections": [
                                {
                                    "operator": "op_a",
                                    "candidates": [
                                        {
                                            "candidate_id": item["candidate"]["id"],
                                            "candidate_sha256": item["candidate"]["code_hash"],
                                        }
                                        for item in manifests
                                    ],
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "baseline candidate|different modification types|Missing modification type",
                ):
                    module.build_batch_smoke(
                        root / "datasets", root / "candidates", root / "batch.zip", selection
                    )

    def test_functional_fallback_requires_v3_slot(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_operator(root, "op_a")
            fallback = self._write_variant(root, "op_a", "fallback", "x + 1", "launch")
            manifest = json.loads(fallback.read_text(encoding="utf-8"))
            manifest["local_performance_evaluation"] = {
                "intended_slot": "v3_functional_fallback_only"
            }
            fallback.write_text(json.dumps(manifest), encoding="utf-8")
            candidate = manifest["candidate"]
            selection = root / "selection.manifest.json"
            selection.write_text(
                json.dumps(
                    {
                        "selections": [
                            {
                                "operator": "op_a",
                                "candidate_id": candidate["id"],
                                "candidate_sha256": candidate["code_hash"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Functional fallback candidate.*slot 3"):
                module.build_batch_smoke(
                    root / "datasets", root / "candidates", root / "batch.zip", selection
                )

    def test_rejects_malformed_local_performance_evaluation(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_operator(root, "op_a")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["local_performance_evaluation"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            selection = self._write_selection_lock(root, ["op_a"])

            with self.assertRaisesRegex(
                ValueError, "Invalid local_performance_evaluation"
            ):
                module.build_batch_smoke(
                    root / "datasets", root / "candidates", root / "batch.zip", selection
                )

    def test_current_admission_requires_independent_holdout_evidence(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_operator(root, "op_a")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("holdout_evaluation")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            selection = self._write_selection_lock(root, ["op_a"])
            with self.assertRaisesRegex(ValueError, "holdout admission.*holdout_required"):
                module.build_batch_smoke(
                    root / "datasets", root / "candidates", root / "batch.zip", selection
                )

    def test_candidate_only_functional_admission_requires_explicit_control_failure(self) -> None:
        module = load_module()
        operator = "_chunk_cumsum_fwd_kernel"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_operator(root, operator)
            evidence = root / "chunk-cumsum-candidate-only.json"
            evidence.write_text(
                json.dumps(
                    {
                        "baseline_status": "failed",
                        "candidate_status": "passed",
                        "status": "candidate_passed_baseline_failed",
                    }
                ),
                encoding="utf-8",
            )
            evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidate_id = manifest["candidate"]["id"]
            candidate_hash = manifest["candidate"]["code_hash"]
            result = {
                "artifact_path": evidence.name,
                "artifact_sha256": evidence_sha256,
                "baseline_status": "failed",
                "candidate_status": "passed",
                "split": "holdout",
                "status": "candidate_passed_baseline_failed",
            }
            manifest["correctness_evaluation"] = {
                "admission_policy_id": module._CANDIDATE_ONLY_FUNCTIONAL_POLICY_ID,
                "blocking_reasons": [],
                "decision": {
                    "candidate_id": candidate_id,
                    "eligible_for_performance": True,
                    "blocking_reasons": [],
                },
                "evidence_scope": "local_ascend_910b4_derived_shape_matrix_not_official",
                "eligible_for_performance": True,
                "results": [result],
                "status": module._CANDIDATE_ONLY_FUNCTIONAL_STATUS,
            }
            manifest["holdout_evaluation"] = {
                "baseline_sha256": "b" * 64,
                "baseline_status": "failed",
                "candidate_id": candidate_id,
                "candidate_sha256": candidate_hash,
                "candidate_status": "passed",
                "case_count": 1,
                "case_signature": "c" * 64,
                "correctness_artifact_path": evidence.name,
                "correctness_artifact_sha256": evidence_sha256,
                "split": "holdout",
                "status": module._CANDIDATE_ONLY_HOLDOUT_STATUS,
                "used_for_search": False,
                "used_in_prompt": False,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            selection = self._write_selection_lock(root, [operator])

            sidecar = module.build_batch_smoke(
                root / "datasets", root / "candidates", root / "batch.zip", selection
            )
            self.assertTrue((root / "batch.zip").is_file())
            self.assertEqual(
                sidecar["selections"][0]["admission_level"],
                "local_ascend_910b4_candidate_only_functional",
            )

    def test_set_k_fp8_candidate_only_admission_is_operator_scoped(self) -> None:
        module = load_module()
        policy = module._SET_K_CANDIDATE_ONLY_FUNCTIONAL_POLICY_ID
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_operator(
                root, "_set_k_and_s_triton_kernel"
            )
            evidence = root / "set-k-fp8.json"
            evidence.write_text(
                json.dumps({"baseline": "failed", "candidate": "passed"}),
                encoding="utf-8",
            )
            evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidate = manifest["candidate"]
            result = {
                "artifact_path": evidence.name,
                "artifact_sha256": evidence_sha256,
                "baseline_status": "failed",
                "candidate_status": "passed",
                "split": "search_visible",
                "status": "baseline_failed",
            }
            manifest["correctness_evaluation"] = {
                "admission_policy_id": policy,
                "blocking_reasons": [],
                "decision": {
                    "candidate_id": candidate["id"],
                    "eligible_for_performance": True,
                    "blocking_reasons": [],
                },
                "evidence_scope": "local_ascend_910b4_derived_shape_matrix_not_official",
                "eligible_for_performance": True,
                "results": [result],
                "status": module._CANDIDATE_ONLY_FUNCTIONAL_STATUS,
            }
            manifest["holdout_evaluation"] = {
                "baseline_sha256": "b" * 64,
                "baseline_status": "failed",
                "candidate_id": candidate["id"],
                "candidate_sha256": candidate["code_hash"],
                "candidate_status": "passed",
                "case_count": 1,
                "case_signature": "c" * 64,
                "correctness_artifact_path": evidence.name,
                "correctness_artifact_sha256": evidence_sha256,
                "split": "holdout",
                "status": module._CANDIDATE_ONLY_HOLDOUT_STATUS,
                "used_for_search": False,
                "used_in_prompt": False,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            selection = self._write_selection_lock(
                root, ["_set_k_and_s_triton_kernel"]
            )

            sidecar = module.build_batch_smoke(
                root / "datasets", root / "candidates", root / "batch.zip", selection
            )
            self.assertEqual(
                sidecar["selections"][0]["admission_level"],
                "local_ascend_910b4_candidate_only_functional",
            )
            self.assertFalse(
                module._has_candidate_only_functional_admission(
                    "op_a", manifest, candidate["id"], (root,)
                )
            )

    def test_selection_lock_rejects_missing_duplicate_and_hash_mismatch(self) -> None:
        module = load_module()
        for label in ("missing", "duplicate", "hash_mismatch"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest_path = self._write_operator(root, "op_a")
                candidate = json.loads(manifest_path.read_text(encoding="utf-8"))["candidate"]
                item = {
                    "operator": "op_a",
                    "candidate_id": candidate["id"],
                    "candidate_sha256": candidate["code_hash"],
                }
                selections = [] if label == "missing" else [item]
                if label == "duplicate":
                    selections.append(dict(item))
                elif label == "hash_mismatch":
                    selections[0]["candidate_sha256"] = "0" * 64
                selection = root / "selection.manifest.json"
                selection.write_text(
                    json.dumps({"selections": selections}), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    ValueError, "missing operators|duplicate operator|identity mismatch"
                ):
                    module.build_batch_smoke(
                        root / "datasets",
                        root / "candidates",
                        root / "batch.zip",
                        selection,
                    )

    def test_historical_rebuild_is_exact_and_never_claims_admission(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._write_operator(root, "op_a")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("import_evaluation")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            selection = self._write_selection_lock(root, ["op_a"])
            historical = self._write_historical_source(root, selection)

            with self.assertRaisesRegex(ValueError, "lacks valid"):
                module.build_batch_smoke(
                    root / "datasets",
                    root / "candidates",
                    root / "default.zip",
                    selection,
                )
            sidecar = module.build_batch_smoke(
                root / "datasets",
                root / "candidates",
                root / "historical-rebuild.zip",
                selection,
                historical_source_manifest=historical,
            )

            self.assertEqual(sidecar["artifact_kind"], "historical-selection-rebuild")
            self.assertEqual(
                sidecar["scoring_intent"],
                "historical-composition-reproduction-not-for-submission",
            )
            self.assertFalse(sidecar["current_admission_claimed"])
            self.assertEqual(
                sidecar["selections"][0]["admission_level"],
                "historical_exact_identity_only_current_admission_not_claimed",
            )

    def test_historical_rebuild_rejects_source_mismatch_and_tampering(self) -> None:
        module = load_module()
        for label in ("selection_mismatch", "artifact_tampering"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write_operator(root, "op_a")
                selection = self._write_selection_lock(root, ["op_a"])
                historical = self._write_historical_source(root, selection)
                if label == "selection_mismatch":
                    document = json.loads(historical.read_text(encoding="utf-8"))
                    document["selections"][0]["candidate_sha256"] = "0" * 64
                    historical.write_text(json.dumps(document), encoding="utf-8")
                else:
                    (root / "historical.zip").write_bytes(b"tampered")

                with self.assertRaisesRegex(
                    ValueError, "do not match|Cannot verify|integrity check"
                ):
                    module.build_batch_smoke(
                        root / "datasets",
                        root / "candidates",
                        root / "batch.zip",
                        selection,
                        historical_source_manifest=historical,
                    )

    def test_generator_persists_import_failure_as_rejected(self) -> None:
        module = load_generator_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator_dir = root / "datasets/op_a"
            operator_dir.mkdir(parents=True)
            (operator_dir / "op_a.py").write_text(
                "def op_a(x):\n    return x\n", encoding="utf-8"
            )
            child = SimpleNamespace(
                code="def op_a(x):\n    return x + 1\n",
                id="candidate",
                generation=1,
                metadata={"mutation_type": "mutation"},
                model_used="fake-model",
            )
            fake_modules = (
                SimpleNamespace(
                    EAConfig=lambda: SimpleNamespace(
                        api_url=None, api_key=None, llm_models=["fake-model"]
                    )
                ),
                SimpleNamespace(
                    GeneticOperators=lambda *_: SimpleNamespace(mutate=lambda _: child),
                    Individual=SimpleNamespace,
                ),
                SimpleNamespace(interface_contract_error=lambda *_: None),
            )
            fake_llm = SimpleNamespace(
                call_history=[{"prompt_sha256": "a" * 64}],
                get_stats=lambda: {"call_count": 1},
            )
            import_failure = SimpleNamespace(
                status="import_error",
                phase="module_import",
                error_type="TypeError",
                error_message="unhashable type: 'dict'",
            )
            static_pass = SimpleNamespace(passed=True, to_dict=lambda: {"passed": True})
            with patch.object(
                module, "_load_official_modules", return_value=fake_modules
            ), patch.object(
                module, "StdlibOpenAIClient", return_value=fake_llm
            ), patch.object(
                module.LocalExecutor, "evaluate", return_value=static_pass
            ), patch.object(
                module, "run_candidate", return_value=import_failure
            ) as runner:
                with self.assertRaisesRegex(ValueError, "failed import gate"):
                    module.generate_candidate(
                        root,
                        root / "datasets",
                        "op_a",
                        operator_dir / "op_a.py",
                        root / "generated",
                        1,
                    )

            request = runner.call_args.args[0]
            self.assertTrue(request.stop_after_import)
            manifest = json.loads(
                (root / "generated/op_a/candidate.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["candidate"]["status"], "rejected")
            self.assertEqual(
                manifest["import_evaluation"],
                {
                    "status": "import_error",
                    "phase": "module_import",
                    "error_type": "TypeError",
                    "error_message": "unhashable type: 'dict'",
                },
            )


if __name__ == "__main__":
    unittest.main()

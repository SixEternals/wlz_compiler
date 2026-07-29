import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, replace
import unittest

from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.official_adapter import (
    BOUND_EVALUATION_KIND,
    BoundOfficialTaskFailure,
    OFFICIAL_FRAMEWORK_COMMIT,
    OfficialTaskFailure,
    adapt_bound_official_evaluation,
    adapt_official_evaluation,
    adapt_official_optimization_result,
    bind_official_task_failures,
    make_exact_official_failure_signature,
    parse_official_task_failures,
)
from wlz_optimizer.schemas import Candidate


@dataclass
class RawOfficialEvaluation:
    success: bool
    execution_time: float
    speedup: float
    fitness: float
    error: str | None = None


class OfficialEvaluationAdapterTests(unittest.TestCase):
    def _artifact(self, candidates: dict[str, tuple[str, str]]) -> tuple[bytes, list[str]]:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for operator, (candidate_id, code) in candidates.items():
                archive.writestr(f"output/{operator}/{operator}_v1.py", code)
                archive.writestr(
                    f"output/{operator}/{operator}_stats.json",
                    json.dumps({"top5_summary": [{"id": candidate_id}]}),
                )
        artifact = buffer.getvalue()
        with zipfile.ZipFile(io.BytesIO(artifact)) as archive:
            return artifact, archive.namelist()

    def _candidate(self) -> Candidate:
        code = "def demo_op(x):\n    return x\n"
        return Candidate(
            id="candidate-1",
            op_name="demo_op",
            code=code,
            code_hash=sha256_text(code),
            parent_ids=["seed"],
            generation=1,
            mutation_kind="mutation",
            model_used="test-model",
            prompt_id="prompt-1",
            status="static_pass",
            score=None,
        )

    def _envelope(self, candidate: Candidate) -> dict:
        return {
            "schema_version": 1,
            "artifact_kind": BOUND_EVALUATION_KIND,
            "operator": candidate.op_name,
            "candidate_id": candidate.id,
            "candidate_code_hash": candidate.code_hash,
            "evaluation": {
                "success": True,
                "execution_time": 2500.0,
                "speedup": 0.6,
                "fitness": 0.6,
                "error": None,
            },
        }

    def test_bound_import_verifies_identity_without_updating_candidate(self) -> None:
        candidate = self._candidate()

        result = adapt_bound_official_evaluation(
            candidate, self._envelope(candidate), baseline_time_us=4000.0
        )

        self.assertTrue(result.metadata["binding_verified"])
        self.assertEqual(result.metadata["bound_candidate_code_hash"], candidate.code_hash)
        self.assertEqual(result.latency_ms, 2.5)
        self.assertIsNone(result.compile_ok)
        self.assertIsNone(result.correctness_ok)
        self.assertIsNone(candidate.score)

    def test_bound_import_rejects_identity_and_source_mismatches(self) -> None:
        candidate = self._candidate()
        for field, value in (
            ("operator", "other_op"),
            ("candidate_id", "other-candidate"),
            ("candidate_code_hash", "0" * 64),
        ):
            with self.subTest(field=field):
                envelope = self._envelope(candidate)
                envelope[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    adapt_bound_official_evaluation(candidate, envelope)

        candidate.code = "def demo_op(x):\n    return x + 1\n"
        with self.assertRaisesRegex(ValueError, "Candidate code"):
            adapt_bound_official_evaluation(candidate, self._envelope(candidate))

    def test_success_preserves_measurement_without_inventing_stage_results(self) -> None:
        raw = RawOfficialEvaluation(True, 2500.0, 0.6, 0.6)

        result = adapt_official_evaluation(
            "candidate-1",
            raw,
            baseline_time_us=4000.0,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.latency_ms, 2.5)
        self.assertEqual(result.baseline_ms, 4.0)
        self.assertEqual(result.speedup, 0.6)
        self.assertIsNone(result.compile_ok)
        self.assertIsNone(result.correctness_ok)
        self.assertIsNone(result.proxy_score)
        self.assertFalse(result.metadata["stage_detail_available"])

    def test_failure_does_not_turn_zero_sentinels_into_measurements(self) -> None:
        raw = {
            "success": False,
            "execution_time": 0.0,
            "speedup": 0.0,
            "fitness": 0.0,
            "error": "Performance test failed",
        }

        result = adapt_official_evaluation("candidate-2", raw)

        self.assertFalse(result.passed)
        self.assertEqual(result.status, "official_evaluation_failed")
        self.assertIsNone(result.latency_ms)
        self.assertIsNone(result.speedup)
        self.assertIsNone(result.compile_ok)
        self.assertIsNone(result.correctness_ok)
        self.assertEqual(result.error_type, "official_evaluation_failed")
        self.assertEqual(result.metadata["official_execution_time_us"], 0.0)

    def test_parse_latest_official_task_failures_without_stage_inference(self) -> None:
        raw = """summary
=== 失败任务 ===
_count_expert_num_tokens tc2 _count_expert_num_tokens_v1: runtime error (Traceback in log) (returncode=0)
_quantize_k_cache_fast_kernel tc2 _quantize_k_cache_fast_kernel_v1: accuracy check failed (AssertionError) (returncode=0)
_quantize_k_cache_fast_kernel tc3 _quantize_k_cache_fast_kernel_v1: accuracy check failed (AssertionError) (returncode=0)
_selective_scan_update_kernel tc1 _selective_scan_update_kernel_v1: accuracy check failed (AssertionError) (returncode=0)
_selective_scan_update_kernel tc2 _selective_scan_update_kernel_v1: accuracy check failed (AssertionError) (returncode=0)
_selective_scan_update_kernel tc3 _selective_scan_update_kernel_v1: accuracy check failed (AssertionError) (returncode=0)
_act_quant_kernel tc3 _act_quant_kernel_v1: runtime error (Traceback in log) (returncode=0)
_set_k_and_s_triton_kernel tc3 _set_k_and_s_triton_kernel_v1: accuracy check failed (AssertionError) (returncode=0)
"""

        failures = parse_official_task_failures(raw)

        self.assertEqual(len(failures), 8)
        self.assertEqual(failures[0].operator, "_count_expert_num_tokens")
        self.assertEqual(failures[0].test_case, "tc2")
        self.assertEqual(failures[0].failure_kind, "runtime_error")
        self.assertEqual(failures[0].returncode, 0)
        self.assertEqual(failures[1].failure_kind, "accuracy_check_failed")
        self.assertNotIn("returncode", failures[1].detail)
        self.assertTrue(all(failure.returncode == 0 for failure in failures))

    def test_parse_task_failures_preserves_unknowns_and_rejects_bad_sections(self) -> None:
        failures = parse_official_task_failures(
            "=== 失败任务 ===\n_demo tc4 _demo_v1: platform-specific failure\n"
        )
        self.assertEqual(failures[0].failure_kind, "unknown")
        self.assertIsNone(failures[0].returncode)
        self.assertEqual(failures[0].detail, "platform-specific failure")

        with self.assertRaisesRegex(ValueError, "no failure section"):
            parse_official_task_failures("评测完成: 63/63\n")
        with self.assertRaisesRegex(ValueError, "Invalid official task failure line"):
            parse_official_task_failures("=== 失败任务 ===\nmalformed failure\n")

    def test_exact_failure_signature_uses_only_stable_bound_fields(self) -> None:
        task_failure = OfficialTaskFailure(
            operator="_op",
            test_case="tc2",
            candidate_variant="_op_v1",
            failure_kind="runtime_error",
            detail="runtime error (Traceback in log)",
            returncode=0,
            raw_line="original line",
        )
        bound = BoundOfficialTaskFailure(
            candidate_id="candidate-1",
            operator="_op",
            candidate_code_hash="a" * 64,
            task_failure=task_failure,
        )
        environment = (
            "coursegrading:contest=1mTsU6jaSZ0:task=14955089:problem=3153461:"
            "assign=47585:observation=20260721-013859"
        )

        signature = make_exact_official_failure_signature(bound, environment)
        diagnostic_change = replace(
            bound,
            candidate_id="candidate-alias",
            task_failure=replace(
                task_failure,
                candidate_variant="platform-alias",
                detail="runtime error (different detail)",
                returncode=17,
                raw_line="different raw line",
            ),
        )

        self.assertEqual(len(signature), 64)
        self.assertEqual(
            signature,
            "d53751f714eb3d045d80de7b7934def39d9a28a6b7df2623fce36eae55f8eef9",
        )
        self.assertEqual(
            make_exact_official_failure_signature(diagnostic_change, environment),
            signature,
        )
        for changed, changed_environment in (
            (replace(bound, candidate_code_hash="b" * 64), environment),
            (
                replace(
                    bound,
                    operator="_other",
                    task_failure=replace(task_failure, operator="_other"),
                ),
                environment,
            ),
            (replace(bound, task_failure=replace(task_failure, test_case="tc3")), environment),
            (
                replace(
                    bound,
                    task_failure=replace(task_failure, failure_kind="accuracy_check_failed"),
                ),
                environment,
            ),
            (bound, environment + "-changed"),
        ):
            self.assertNotEqual(
                make_exact_official_failure_signature(changed, changed_environment),
                signature,
            )

        unknown = replace(
            bound, task_failure=replace(task_failure, failure_kind="unknown")
        )
        self.assertEqual(
            len(make_exact_official_failure_signature(unknown, environment)), 64
        )
        for invalid_environment in ("", "unknown", "default", "x"):
            with self.subTest(env=invalid_environment), self.assertRaisesRegex(
                ValueError, "environment fingerprint"
            ):
                make_exact_official_failure_signature(bound, invalid_environment)
        with self.assertRaisesRegex(ValueError, "Unsupported official failure kind"):
            make_exact_official_failure_signature(
                replace(bound, task_failure=replace(task_failure, failure_kind="compile_error")),
                environment,
            )

    def test_bind_task_failures_resolves_overlay_and_base_candidates(self) -> None:
        failures = parse_official_task_failures(
            "=== 失败任务 ===\n"
            "_replaced tc1 _replaced_v1: runtime error (returncode=0)\n"
            "_preserved tc2 _preserved_v1: accuracy check failed (returncode=0)\n"
        )
        replacement_code = "replacement code"
        preserved_code = "preserved code"
        artifact, archive_entries = self._artifact(
            {
                "_replaced": ("replacement-candidate", replacement_code),
                "_preserved": ("base-candidate", preserved_code),
            }
        )
        artifact_sha = hashlib.sha256(artifact).hexdigest()
        base = {
            "artifact_sha256": "a" * 64,
            "selections": [
                {
                    "operator": "_replaced",
                    "candidate_id": "old-candidate",
                    "candidate_sha256": "d" * 64,
                },
                {
                    "operator": "_preserved",
                    "candidate_id": "base-candidate",
                    "candidate_sha256": sha256_text(preserved_code),
                }
            ],
        }
        overlay = {
            "base_artifact": {"sha256": "a" * 64},
            "artifact_sha256": artifact_sha,
            "archive_entries": archive_entries,
            "replacements": [
                {
                    "operator": "_replaced",
                    "candidate_id": "replacement-candidate",
                    "candidate_sha256": sha256_text(replacement_code),
                }
            ],
        }

        bound = bind_official_task_failures(
            failures,
            overlay,
            artifact_sha,
            artifact_bytes=artifact,
            base_manifest=base,
        )

        self.assertEqual(
            [record.candidate_id for record in bound],
            ["replacement-candidate", "base-candidate"],
        )
        self.assertEqual(bound[0].candidate_code_hash, sha256_text(replacement_code))
        self.assertEqual(bound[1].operator, "_preserved")
        self.assertIs(bound[1].task_failure, failures[1])

    def test_bind_task_failures_rejects_ambiguous_identity(self) -> None:
        failure = parse_official_task_failures(
            "=== 失败任务 ===\n_op tc1 _wrong_v1: runtime error\n"
        )
        code = "candidate code"
        artifact, archive_entries = self._artifact({"_op": ("candidate", code)})
        artifact_sha = hashlib.sha256(artifact).hexdigest()
        manifest = {
            "artifact_sha256": artifact_sha,
            "archive_entries": archive_entries,
            "selections": [
                {
                    "operator": "_op",
                    "candidate_id": "candidate",
                    "candidate_sha256": sha256_text(code),
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "variant mismatch"):
            bind_official_task_failures(
                failure, manifest, artifact_sha, artifact_bytes=artifact
            )

        manifest["selections"][0]["candidate_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            bind_official_task_failures(
                failure, manifest, artifact_sha, artifact_bytes=artifact
            )

        overlay = {"artifact_sha256": artifact_sha, "base_artifact": {"sha256": "a" * 64}}
        with self.assertRaisesRegex(ValueError, "requires its base manifest"):
            bind_official_task_failures(
                failure, overlay, artifact_sha, artifact_bytes=artifact
            )

        manifest["selections"][0]["candidate_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "does not match expected artifact"):
            bind_official_task_failures(
                failure, manifest, "f" * 64, artifact_bytes=artifact
            )

        valid_base = {
            "artifact_sha256": "a" * 64,
            "selections": manifest["selections"],
        }
        valid_overlay = {
            "artifact_sha256": artifact_sha,
            "base_artifact": {"sha256": "a" * 64},
            "archive_entries": manifest["archive_entries"],
            "replacements": manifest["selections"],
        }
        for bad_base, message in (
            ({"selections": manifest["selections"]}, "base manifest SHA-256"),
            ({**valid_base, "artifact_sha256": "b" * 64}, "base manifest hash mismatch"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                bind_official_task_failures(
                    failure,
                    valid_overlay,
                    artifact_sha,
                    artifact_bytes=artifact,
                    base_manifest=bad_base,
                )

        unknown_replacement = {
            **valid_overlay,
            "replacements": [
                {**manifest["selections"][0], "operator": "_other"}
            ],
        }
        with self.assertRaisesRegex(ValueError, "subset"):
            bind_official_task_failures(
                failure,
                unknown_replacement,
                artifact_sha,
                artifact_bytes=artifact,
                base_manifest=valid_base,
            )
        with self.assertRaisesRegex(ValueError, "subset"):
            bind_official_task_failures(
                failure,
                {**valid_overlay, "selections": manifest["selections"]},
                artifact_sha,
                artifact_bytes=artifact,
                base_manifest=valid_base,
            )

        with self.assertRaisesRegex(ValueError, "cannot contain replacements"):
            bind_official_task_failures(
                failure,
                {**manifest, "selections": [], "replacements": manifest["selections"]},
                artifact_sha,
                artifact_bytes=artifact,
            )
        correct_failure = parse_official_task_failures(
            "=== 失败任务 ===\n_op tc1 _op_v1: runtime error\n"
        )
        for entries, message in (
            (manifest["archive_entries"] * 2, "duplicate archive_entries"),
            ([], "ZIP entries do not match"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                bind_official_task_failures(
                    correct_failure,
                    {**manifest, "archive_entries": entries},
                    artifact_sha,
                    artifact_bytes=artifact,
                )

        manifest["selections"][0]["candidate_sha256"] = sha256_text(code)
        with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
            bind_official_task_failures(
                correct_failure,
                manifest,
                artifact_sha,
                artifact_bytes=artifact + b"tampered",
            )

        manifest["selections"][0]["candidate_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            bind_official_task_failures(
                correct_failure, manifest, artifact_sha, artifact_bytes=artifact
            )

        bad_stats, bad_stats_entries = self._artifact({"_op": ("other", code)})
        bad_stats_sha = hashlib.sha256(bad_stats).hexdigest()
        manifest["selections"][0]["candidate_sha256"] = sha256_text(code)
        with self.assertRaisesRegex(ValueError, "stats identity mismatch"):
            bind_official_task_failures(
                correct_failure,
                {
                    **manifest,
                    "artifact_sha256": bad_stats_sha,
                    "archive_entries": bad_stats_entries,
                },
                bad_stats_sha,
                artifact_bytes=bad_stats,
            )

        duplicate_stats = io.BytesIO()
        with zipfile.ZipFile(duplicate_stats, "w") as archive:
            archive.writestr("output/_op/_op_v1.py", code)
            archive.writestr(
                "output/_op/_op_stats.json",
                '{"top5_summary":[{"id":"wrong","id":"candidate"}]}',
            )
        duplicate_stats_bytes = duplicate_stats.getvalue()
        duplicate_stats_sha = hashlib.sha256(duplicate_stats_bytes).hexdigest()
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            bind_official_task_failures(
                correct_failure,
                {
                    **manifest,
                    "artifact_sha256": duplicate_stats_sha,
                    "archive_entries": bad_stats_entries,
                },
                duplicate_stats_sha,
                artifact_bytes=duplicate_stats_bytes,
            )


class OfficialOptimizationAdapterTests(unittest.TestCase):
    def _raw_result(self, count: int = 2) -> dict:
        return {
            "best_code": "code-0",
            "best_fitness": 0.8,
            "speedup": 0.8,
            "generations": 3,
            "time_elapsed": 10.0,
            "llm_stats": {"call_count": 4},
            "top5_codes": [
                {
                    "code": f"code-{index}",
                    "fitness": 0.8 - index * 0.1,
                    "generation": index,
                    "id": f"official-{index}",
                }
                for index in range(count)
            ],
        }

    def test_top5_maps_available_provenance_and_marks_missing_fields(self) -> None:
        candidates = adapt_official_optimization_result("demo_op", self._raw_result())

        self.assertEqual(len(candidates), 2)
        first = candidates[0]
        self.assertEqual(first.id, "official-0")
        self.assertEqual(first.op_name, "demo_op")
        self.assertEqual(first.code_hash, sha256_text("code-0"))
        self.assertEqual(first.generation, 0)
        self.assertEqual(first.score, 0.8)
        self.assertEqual(first.metadata["official_framework_commit"], OFFICIAL_FRAMEWORK_COMMIT)
        self.assertFalse(first.metadata["provenance_complete"])
        self.assertIn("parent_ids", first.metadata["missing_provenance_fields"])

    def test_more_than_five_candidates_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "more than five"):
            adapt_official_optimization_result("demo_op", self._raw_result(count=6))


if __name__ == "__main__":
    unittest.main()

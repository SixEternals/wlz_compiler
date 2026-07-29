import json
import tempfile
import unittest
from pathlib import Path

from wlz_optimizer.cache import EvaluationCache, OfficialFailureHistory
from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.official_adapter import (
    OFFICIAL_EXECUTOR_KIND,
    BoundOfficialTaskFailure,
    OfficialTaskFailure,
)
from wlz_optimizer.prompt_context import EvidenceView, PromptContextProjector
from wlz_optimizer.schemas import Candidate, EvaluationResult, ShapeObservation


ENV = (
    "coursegrading:contest=1mTsU6jaSZ0:task=14955089:problem=3153461:"
    "assign=47585:observation=20260721-013859"
)


class PromptContextTests(unittest.TestCase):
    def setUp(self) -> None:
        code = "def demo_op(x):\n    return x\n"
        self.parent = Candidate(
            id="parent-1", op_name="demo_op", code=code,
            code_hash=sha256_text(code), parent_ids=[], generation=3,
            mutation_kind="mutation", model_used="deepseek", prompt_id="p1",
            status="created", score=0.75,
        )

    def _result(self, **changes: object) -> EvaluationResult:
        data = dict(
            candidate_id=self.parent.id, executor=OFFICIAL_EXECUTOR_KIND, status="failed",
            passed=False, correctness_ok=None, compile_ok=False,
            latency_ms=None, baseline_ms=None, speedup=None, proxy_score=0.9,
            error_type="compile_error", error_message="tc7 SECRET raw log",
            metadata={"secret": "do-not-expose"},
        )
        data.update(changes)
        return EvaluationResult(**data)

    def test_projector_is_bounded_and_strips_raw_diagnostics(self) -> None:
        shapes = (
            ShapeObservation(
                op_name="demo_op",
                case_id="case-secret-1",
                tensor_shapes={"input-secret": [2, 64], "mask-secret": [2, None]},
                tensor_dtypes={
                    "input-secret": "torch.float16",
                    "mask-secret": "torch.bool",
                },
                source="public-test-secret",
            ),
        )
        evidence = EvidenceView(
            self.parent, ENV,
            (
                self._result(
                    executor="local_proxy", speedup=99.0, error_type="runtime_error"
                ),
                self._result(
                    passed=True, compile_ok=True, correctness_ok=True, speedup=1.2,
                    latency_ms=0.75, error_type=None, error_message=None,
                ),
            ),
            shape_observations=shapes,
        )
        context = PromptContextProjector().project(evidence)
        encoded = json.dumps(context.to_dict(), sort_keys=True)
        self.assertEqual(context.parent_code_hash, self.parent.code_hash)
        self.assertEqual(context.generation, 3)
        self.assertEqual(context.evaluation_count, 2)
        self.assertEqual(context.evaluation_pass_count, 1)
        self.assertEqual(context.compile_counts, (1, 1, 0))
        self.assertEqual(context.correctness_counts, (1, 0, 1))
        self.assertEqual(context.failure_category_counts, (("runtime_error", 1),))
        self.assertEqual(context.observed_speedups, (1.2,))
        self.assertEqual(context.shape_observation_count, 1)
        self.assertEqual(context.tensor_rank_counts, ((2, 2),))
        self.assertEqual(context.dtype_family_counts, (("bool", 1), ("float", 1)))
        self.assertEqual(context.unknown_dimension_count, 1)
        self.assertEqual(context.official_performance_count, 1)
        self.assertEqual(context.official_speedup_best, 1.2)
        self.assertEqual(context.official_speedup_median, 1.2)
        self.assertEqual(context.official_speedup_latest, 1.2)
        self.assertEqual(context.official_latency_ms_best, 0.75)
        self.assertEqual(context.sanitization_version, "prompt-context-sanitization-v2")
        self.assertNotIn("SECRET", encoded)
        self.assertNotIn("tc7", encoded)
        self.assertNotIn(ENV, encoded)
        self.assertNotIn("demo_op", encoded)
        self.assertNotIn("do-not-expose", encoded)
        self.assertNotIn("proxy_score", encoded)
        self.assertNotIn("case-secret", encoded)
        self.assertNotIn("input-secret", encoded)
        self.assertNotIn("public-test-secret", encoded)
        self.assertNotIn("[2, 64]", encoded)
        self.assertEqual(context, PromptContextProjector().project(evidence))

        many = tuple(
            self._result(
                passed=True, compile_ok=True, speedup=float(index),
                error_type=None, error_message=None,
            )
            for index in range(10)
        )
        bounded = PromptContextProjector().project(EvidenceView(self.parent, ENV, many))
        self.assertEqual(bounded.observed_speedups, tuple(float(i) for i in range(2, 10)))

    def test_cache_and_official_history_are_exactly_parent_and_environment_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = EvaluationCache(Path(tmp) / "cache.jsonl")
            cached = self._result(
                passed=True, compile_ok=True, correctness_ok=None, speedup=0.8,
                error_type=None, error_message=None,
            )
            cache.put(self.parent, cached, OFFICIAL_EXECUTOR_KIND, ENV)
            replayed = cache.get(self.parent, OFFICIAL_EXECUTOR_KIND, ENV)
            self.assertIsNotNone(replayed)

            task = OfficialTaskFailure(
                "demo_op", "tc2", "demo_op_v1", "runtime_error", "runtime error", 1,
                "demo_op tc2 demo_op_v1: runtime error (returncode=1)",
            )
            history = OfficialFailureHistory(Path(tmp) / "failures.jsonl")
            history.append(
                BoundOfficialTaskFailure("parent-1", "demo_op", self.parent.code_hash, task),
                ENV, "20260721-013859",
            )

            context = PromptContextProjector().project(
                EvidenceView(
                    self.parent, ENV, (replayed,), tuple(history.entries.values())
                )
            )
        self.assertEqual(context.evaluation_count, 1)
        self.assertEqual(context.failure_category_counts, (("runtime_error", 1),))
        self.assertTrue(context.environment_bound)

    def test_unknown_official_failure_kind_coarsens_to_other_and_renders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = OfficialTaskFailure(
                "demo_op", "tc3", "demo_op_v1", "unknown", "mystery failure", 1,
                "demo_op tc3 demo_op_v1: mystery failure (returncode=1)",
            )
            history = OfficialFailureHistory(Path(tmp) / "failures.jsonl")
            history.append(
                BoundOfficialTaskFailure("parent-1", "demo_op", self.parent.code_hash, task),
                ENV, "20260721-013859",
            )
            context = PromptContextProjector().project(
                EvidenceView(self.parent, ENV, (), tuple(history.entries.values()))
            )
        self.assertEqual(context.failure_category_counts, (("other", 1),))

        from scripts.generate_official_candidate import _load_official_modules

        _, operators, _ = _load_official_modules(
            (Path(__file__).resolve().parents[1] / "work" / "official_triton_agent").resolve()
        )
        rendered = operators.render_prompt_context(context.to_dict())
        self.assertTrue(any("other=1" in line for line in rendered))

    def test_syntax_error_parent_reports_no_source_access_counts(self) -> None:
        code = "def broken(:\n"
        parent = Candidate(
            id="parent-broken", op_name="demo_op", code=code,
            code_hash=sha256_text(code), parent_ids=[], generation=1,
            mutation_kind="mutation", model_used="deepseek", prompt_id="p4",
            status="created", score=None,
        )
        context = PromptContextProjector().project(EvidenceView(parent, ENV, ()))
        self.assertEqual(context.source_access_counts, ())

    def test_source_access_and_performance_summaries_are_deterministic(self) -> None:
        code = (
            "def demo_op(x, y):\n"
            "    value = tl.load(x)\n"
            "    tl.store(y, value)\n"
            "    tl.atomic_add(y, value)\n"
            "    return tl.trans(value)\n"
        )
        parent = Candidate(
            id="parent-access",
            op_name="demo_op",
            code=code,
            code_hash=sha256_text(code),
            parent_ids=[],
            generation=1,
            mutation_kind="mutation",
            model_used="deepseek",
            prompt_id="p2",
            status="created",
            score=None,
        )
        results = (
            self._result(
                candidate_id=parent.id,
                passed=True,
                compile_ok=True,
                correctness_ok=True,
                speedup=1.0,
                latency_ms=2.0,
                error_type=None,
                error_message=None,
            ),
            self._result(
                candidate_id=parent.id,
                passed=True,
                compile_ok=True,
                correctness_ok=True,
                speedup=2.0,
                latency_ms=1.0,
                error_type=None,
                error_message=None,
            ),
        )
        context = PromptContextProjector().project(EvidenceView(parent, ENV, results))
        self.assertEqual(
            context.source_access_counts,
            (
                ("loads", 1),
                ("stores", 1),
                ("atomics", 1),
                ("block_pointers", 0),
                ("transposes", 1),
            ),
        )
        self.assertEqual(context.official_performance_count, 2)
        self.assertEqual(context.official_speedup_best, 2.0)
        self.assertEqual(context.official_speedup_median, 1.5)
        self.assertEqual(context.official_speedup_latest, 2.0)
        self.assertEqual(context.official_latency_ms_best, 1.0)

    def test_summary_buckets_high_rank_and_ignores_non_triton_accessors(self) -> None:
        code = "def demo_op(x):\n    other.load(x)\n    return tl.load(x)\n"
        parent = Candidate(
            id="parent-bucket",
            op_name="demo_op",
            code=code,
            code_hash=sha256_text(code),
            parent_ids=[],
            generation=1,
            mutation_kind="mutation",
            model_used="deepseek",
            prompt_id="p3",
            status="created",
            score=None,
        )
        observations = tuple(
            ShapeObservation(
                op_name="demo_op",
                case_id=f"case-{index}",
                tensor_shapes={"x": [1] * 20},
            )
            for index in range(2)
        )
        context = PromptContextProjector().project(
            EvidenceView(parent, ENV, (), shape_observations=observations)
        )
        self.assertEqual(context.tensor_rank_counts, ((16, 2),))
        self.assertEqual(
            context.source_access_counts,
            (
                ("loads", 1),
                ("stores", 0),
                ("atomics", 0),
                ("block_pointers", 0),
                ("transposes", 0),
            ),
        )

    def test_mismatched_evaluation_fails_closed(self) -> None:
        other = self._result(candidate_id="other")
        with self.assertRaises(ValueError):
            PromptContextProjector().project(EvidenceView(self.parent, None, (other,)))

        mismatched_shape = ShapeObservation(
            op_name="other_op",
            case_id="case-1",
            tensor_shapes={"x": [4]},
        )
        with self.assertRaises(ValueError):
            PromptContextProjector().project(
                EvidenceView(
                    self.parent,
                    None,
                    (),
                    shape_observations=(mismatched_shape,),
                )
            )


if __name__ == "__main__":
    unittest.main()

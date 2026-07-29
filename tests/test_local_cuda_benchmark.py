"""Focused tests for correctness-gated local CUDA proxy timing."""

import importlib.util
import unittest

from wlz_optimizer.correctness import decide_candidate_correctness
from wlz_optimizer.local_cuda_benchmark import (
    CudaBenchmarkConfig,
    benchmark_cuda_callable,
    compare_cuda_callables,
)
from wlz_optimizer.schemas import CorrectnessCaseResult, CorrectnessErrorSummary


SIG = "a" * 64


def decision(status="passed"):
    result = CorrectnessCaseResult(
        candidate_id="candidate-1",
        case_id="case-a",
        case_signature=SIG,
        oracle_policy_id="exact-v1",
        oracle_status=status,
        error_summary=(
            CorrectnessErrorSummary(mismatch_kind="value", mismatch_count=1, compared_count=1)
            if status == "failed"
            else None
        ),
    )
    return decide_candidate_correctness("candidate-1", [SIG], [result])


class LocalCudaBenchmarkTests(unittest.TestCase):
    def test_blocked_correctness_never_calls_callable(self):
        calls = []
        result = benchmark_cuda_callable(
            "candidate-1", SIG, decision("failed"), lambda: calls.append("called")
        )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.executor, "local_cuda_proxy")
        self.assertEqual(calls, [])
        self.assertIsNone(result.local_latency_ms)

    def test_unconfigured_cuda_is_explicit(self):
        fake_torch = type(
            "NoCudaTorch",
            (),
            {"cuda": type("Cuda", (), {"is_available": staticmethod(lambda: False)})()},
        )()
        result = benchmark_cuda_callable(
            "candidate-1",
            SIG,
            decision(),
            lambda: None,
            torch_module=fake_torch,
        )

        self.assertEqual(result.status, "not_configured")
        self.assertEqual(result.samples_ms, ())

    def test_broken_available_cuda_is_not_configured(self):
        fake_torch = type(
            "BrokenCudaTorch",
            (),
            {"cuda": type("Cuda", (), {"is_available": staticmethod(lambda: True)})()},
        )()
        result = benchmark_cuda_callable(
            "candidate-1",
            SIG,
            decision(),
            lambda: None,
            torch_module=fake_torch,
        )

        self.assertEqual(result.status, "not_configured")
        self.assertIn("CUDA backend unavailable", result.message)

    def test_config_rejects_invalid_runs_and_percentile(self):
        with self.assertRaisesRegex(ValueError, "measurement_runs"):
            CudaBenchmarkConfig(measurement_runs=0)
        with self.assertRaisesRegex(ValueError, "percentile"):
            CudaBenchmarkConfig(percentile=101)

    def test_interleaved_comparison_alternates_pair_order(self):
        order = []

        class Event:
            def record(self, stream):
                self.stream = stream

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 1.0

        cuda = type(
            "Cuda",
            (),
            {
                "is_available": staticmethod(lambda: True),
                "current_stream": staticmethod(
                    lambda: type(
                        "Stream",
                        (),
                        {"device": type("Device", (), {"index": 0})()},
                    )()
                ),
                "current_device": staticmethod(lambda: 0),
                "get_device_name": staticmethod(lambda index: "fake-cuda"),
                "get_device_capability": staticmethod(lambda index: (9, 0)),
                "synchronize": staticmethod(lambda: None),
                "Event": staticmethod(lambda enable_timing: Event()),
            },
        )()
        fake_torch = type(
            "FakeTorch",
            (),
            {
                "cuda": cuda,
                "__version__": "fake",
                "version": type("Version", (), {"cuda": "fake"})(),
            },
        )()
        result = compare_cuda_callables(
            "candidate-1",
            SIG,
            decision(),
            lambda: order.append("baseline"),
            lambda: order.append("candidate"),
            torch_module=fake_torch,
            config=CudaBenchmarkConfig(warmup_runs=0, measurement_runs=4),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(
            order,
            [
                "baseline", "candidate",
                "candidate", "baseline",
                "baseline", "candidate",
                "candidate", "baseline",
            ],
        )
        self.assertEqual(result.candidate_over_baseline_latency_ratio, 1.0)
        self.assertEqual(len(result.baseline_samples_ms), 4)

    def test_explicit_stream_controls_context_events_and_fingerprint_device(self):
        calls = []

        class Stream:
            device = type("Device", (), {"index": 1})()

        timing_stream = Stream()

        class StreamContext:
            def __enter__(self):
                calls.append("enter")

            def __exit__(self, *args):
                calls.append("exit")

        class Event:
            def record(self, stream):
                calls.append(("record", stream))

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 1.0

        cuda = type(
            "Cuda",
            (),
            {
                "is_available": staticmethod(lambda: True),
                "current_device": staticmethod(lambda: 0),
                "get_device_name": staticmethod(
                    lambda index: calls.append(("name", index)) or "fake-cuda"
                ),
                "get_device_capability": staticmethod(
                    lambda index: calls.append(("capability", index)) or (9, 0)
                ),
                "stream": staticmethod(lambda stream: StreamContext()),
                "synchronize": staticmethod(lambda: None),
                "Event": staticmethod(lambda enable_timing: Event()),
            },
        )()
        fake_torch = type(
            "FakeTorch",
            (),
            {
                "cuda": cuda,
                "__version__": "fake",
                "version": type("Version", (), {"cuda": "fake"})(),
            },
        )()

        result = benchmark_cuda_callable(
            "candidate-1",
            SIG,
            decision(),
            lambda: calls.append("call"),
            torch_module=fake_torch,
            stream=timing_stream,
            config=CudaBenchmarkConfig(warmup_runs=0, measurement_runs=1),
        )

        self.assertEqual(result.status, "completed")
        self.assertIn(("name", 1), calls)
        self.assertIn(("capability", 1), calls)
        self.assertEqual(calls.count(("record", timing_stream)), 2)
        self.assertLess(calls.index("enter"), calls.index("call"))
        self.assertLess(calls.index("call"), calls.index("exit"))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None
        and __import__("torch").cuda.is_available(),
        "CUDA benchmark requires a compatible GPU environment",
    )
    def test_real_cuda_events_return_samples_and_percentiles(self):
        import torch

        tensor = torch.randn((1024, 1024), device="cuda")
        result = benchmark_cuda_callable(
            "candidate-1",
            SIG,
            decision(),
            lambda: torch.relu(tensor),
            torch_module=torch,
            config=CudaBenchmarkConfig(warmup_runs=2, measurement_runs=5, percentile=95),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.samples_ms), 5)
        self.assertGreaterEqual(result.local_latency_ms, 0.0)
        self.assertGreaterEqual(result.percentile_latency_ms, result.local_latency_ms)
        self.assertEqual(len(result.environment_fingerprint), 16)

        comparison = compare_cuda_callables(
            "candidate-1",
            SIG,
            decision(),
            lambda: torch.relu(tensor),
            lambda: torch.neg(tensor),
            torch_module=torch,
            config=CudaBenchmarkConfig(warmup_runs=2, measurement_runs=5),
        )
        self.assertEqual(comparison.status, "completed")
        self.assertEqual(len(comparison.baseline_samples_ms), 5)
        self.assertGreater(comparison.candidate_over_baseline_latency_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()

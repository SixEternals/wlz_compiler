import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/probe_set_k_and_s_raw_layout.py"
    spec = importlib.util.spec_from_file_location("set_k_raw_layout_probe_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROBE = load_module()


class SetKAndSRawLayoutTests(unittest.TestCase):
    def test_page_stride_and_boundaries_are_raw_byte_addresses(self) -> None:
        baseline = PROBE.inspect_locs("baseline", (0, 64, 128, 192))
        control = PROBE.inspect_locs("fp16_row_stride_control", (63, 64, 127, 128))

        self.assertEqual([item["k_raw_interval"][0] for item in baseline["addresses"]], [0, 16896, 33792, 50688])
        self.assertTrue(baseline["summary"]["any_k_page_mismatch"])
        self.assertTrue(baseline["summary"]["any_out_of_buffer"])
        self.assertEqual([item["k_raw_interval"][0] for item in control["addresses"]], [16128, 8448, 24576, 16896])
        self.assertEqual([item["scale_raw_interval"][0] for item in control["addresses"]], [8444, 16640, 16892, 25088])

    def test_control_still_detects_scale_overlap_and_oob(self) -> None:
        control = PROBE.inspect_locs("fp16_row_stride_control", (0, 32, 255))
        self.assertFalse(control["addresses"][0]["k_overlaps_scale_region"])
        self.assertTrue(control["addresses"][1]["k_overlaps_scale_region"])
        self.assertIn("k_out_of_buffer", control["addresses"][2]["issues"])

    def test_report_is_source_bound_and_non_admission(self) -> None:
        source = ROOT / "work/official_triton_agent/datasets/_set_k_and_s_triton_kernel/_set_k_and_s_triton_kernel.py"
        report = PROBE.build_report(source)

        self.assertEqual(report["source"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(report["semantic_conclusion"], "unknown")
        self.assertEqual(report["candidate_admission"], "not_applicable")
        self.assertEqual([item["name"] for item in report["scenarios"]], ["all_pages", "page_boundaries", "permuted_pages", "static_conflicts"])

    def test_baseline_control_requires_verified_modeled_writes(self) -> None:
        source = ROOT / "work/official_triton_agent/datasets/_set_k_and_s_triton_kernel/_set_k_and_s_triton_kernel.py"
        locs = (0, 64, 128, 192)
        result = {
            "status": "passed",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "locs": list(locs),
            "device_name": "Ascend910B4-1",
            "k_writes_match_modeled_offsets": True,
            "scale_writes_match_modeled_offsets": True,
            "prefix_guard_unchanged": True,
            "suffix_guard_changed": True,
            "writes_beyond_logical_buffer": True,
            "unexpected_changed_byte_count": 0,
            "k_raw_intervals": [[0, 256]],
            "scale_raw_intervals": [[8192, 8196]],
            "storage_sha256": "a" * 64,
        }
        summary = PROBE._validate_control_result(result, source, locs)
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["k_writes_match_modeled_offsets"])
        self.assertTrue(summary["scale_writes_match_modeled_offsets"])
        self.assertTrue(summary["writes_beyond_logical_buffer"])

        report = PROBE.build_report(source, baseline_control={"status": "passed"})
        self.assertIn("observes raw writes only", report["limitations"][0])

        result["unexpected_changed_byte_count"] = 1
        with self.assertRaisesRegex(ValueError, "did not match"):
            PROBE._validate_control_result(result, source, locs)

    def test_cli_renders_control_as_non_admission_evidence(self) -> None:
        control = {"status": "passed", "scenarios": []}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            with redirect_stdout(io.StringIO()):
                with (
                    patch.object(PROBE, "run_baseline_control", return_value=control),
                    patch.object(
                        sys,
                        "argv",
                        [
                            "probe_set_k_and_s_raw_layout.py",
                            "--baseline-control-python",
                            "/unused/ascend-python",
                            "--output",
                            str(output),
                        ],
                    ),
                ):
                    self.assertEqual(PROBE.main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["baseline_control"], control)
        self.assertIn("observes raw writes only", report["limitations"][0])


if __name__ == "__main__":
    unittest.main()

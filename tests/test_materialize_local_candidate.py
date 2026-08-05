"""Tests for deterministic local candidate materialization."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.materialize_local_candidate import materialize
from scripts.materialize_local_candidate import ROOT


class MaterializeLocalCandidateTests(unittest.TestCase):
    def test_materializes_hash_bound_nonbaseline_candidate(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            baseline = root / "baseline.py"
            source = root / "variant.py"
            output = root / "output"
            baseline.write_text("def op(x):\n    return x\n", encoding="utf-8")
            source.write_text("def op(x):\n    return x + 1\n", encoding="utf-8")

            candidate, manifest = materialize("op", baseline, source, output)

            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(candidate.read_bytes(), source.read_bytes())
            self.assertEqual(data["candidate"]["id"], candidate.stem)
            import hashlib

            self.assertEqual(
                data["candidate"]["code_hash"],
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
            )
            self.assertEqual(data["candidate"]["status"], "static_pass")
            self.assertIsNone(data["candidate"]["score"])

    def test_rejects_baseline_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            baseline = root / "baseline.py"
            baseline.write_text("x = 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                materialize("op", baseline, baseline, root / "output")
            source = root / "variant.py"
            source.write_text("x = 2\n", encoding="utf-8")
            materialize("op", baseline, source, root / "output")
            with self.assertRaises(FileExistsError):
                materialize("op", baseline, source, root / "output")

    def test_exact_text_rewrite_can_derive_from_baseline(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            baseline = root / "baseline.py"
            baseline.write_text("def op():\n    return 1\n", encoding="utf-8")
            candidate, manifest = materialize(
                "op", baseline, baseline, root / "output",
                mutation_kind="launch_parameter_probe",
                replace_old="return 1",
                replace_new="return 2",
            )
            self.assertEqual(candidate.read_text(encoding="utf-8").splitlines()[-1], "    return 2")
            self.assertEqual(json.loads(manifest.read_text())["candidate"]["mutation_kind"], "launch_parameter_probe")

    def test_manifest_records_official_contract_preflight(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            baseline = root / "baseline.py"
            source = root / "variant.py"
            output = root / "output"
            baseline.write_text("def op(x):\n    return x\n", encoding="utf-8")
            source.write_text("def op(x):\n    return x + 1\n", encoding="utf-8")

            _, manifest = materialize("op", baseline, source, output)

            static = json.loads(
                manifest.read_text(encoding="utf-8")
            )["static_evaluation"]
            self.assertEqual(static["executor"], "official-contract-preflight")
            self.assertEqual(static["status"], "interface_contract_pass")
            self.assertEqual(static["case_count"], 0)

    def test_rejects_timing_surface_drift_before_writing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            baseline = root / "_op.py"
            source = root / "variant.py"
            output = root / "output"
            test = root / "test__op_1.py"
            baseline_text = (
                "import torch\n"
                "import triton\n"
                "import triton.language as tl\n\n"
                "@triton.jit\n"
                "def kernel(x, BLOCK_SIZE: tl.constexpr):\n"
                "    return\n\n"
                "def op(x):\n"
                "    kernel[(1,)](x, BLOCK_SIZE=128)\n"
                "    return x\n"
            )
            baseline.write_text(baseline_text, encoding="utf-8")
            source.write_text(
                baseline_text.replace(
                    "    kernel[(1,)]",
                    "    torch.cumsum(x, dim=0)\n    kernel[(1,)]",
                ),
                encoding="utf-8",
            )
            test.write_text("from _op import op\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "static preflight failed"):
                materialize("_op", baseline, source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

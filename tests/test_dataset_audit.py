import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.dataset_audit import audit_dataset
from wlz_optimizer.schemas import (
    ShapeConstraint,
    ShapeContract,
    SymbolicDimension,
    TensorAxisBinding,
)


class DatasetAuditTests(unittest.TestCase):
    def test_audit_dataset_reports_seed_signature_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "datasets"
            op_dir = input_dir / "demo_op"
            op_dir.mkdir(parents=True)
            (op_dir / "demo_op.py").write_text(
                "\n".join(
                    [
                        "import triton",
                        "import triton.language as tl",
                        "@triton.jit",
                        "def kernel(x):",
                        "    return",
                        "def demo_op(x):",
                        "    return x",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (op_dir / "demo_op_1.py").write_text(
                "\n".join(
                    [
                        "from vllm.triton_utils import tl, triton",
                        "@triton.jit",
                        "def kernel(x):",
                        "    return",
                        "def wrong_name(x):",
                        "    return x",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (op_dir / "test_demo_op_1.py").write_text(
                "from demo_op import demo_op\n",
                encoding="utf-8",
            )

            report = audit_dataset(input_dir)
            self.assertEqual(report["summary"]["operator_count"], 1)
            self.assertEqual(report["summary"]["seed_count"], 2)
            self.assertEqual(report["summary"]["signature_fail_count"], 1)
            self.assertEqual(report["summary"]["import_style_counts"]["standard_triton"], 1)
            self.assertEqual(report["summary"]["import_style_counts"]["vllm.triton_utils"], 1)

    def test_audit_script_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "datasets"
            output_dir = tmp_path / "audit"
            op_dir = input_dir / "demo_op"
            op_dir.mkdir(parents=True)
            (op_dir / "demo_op.py").write_text(
                "\n".join(
                    [
                        "import triton",
                        "import triton.language as tl",
                        "@triton.jit",
                        "def kernel(x):",
                        "    return",
                        "def demo_op(x):",
                        "    return x",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (op_dir / "test_demo_op_1.py").write_text(
                "from demo_op import demo_op\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_dataset.py"),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            json_path = output_dir / "dataset_audit.json"
            md_path = output_dir / "dataset_audit.md"
            self.assertTrue(json_path.is_file())
            self.assertTrue(md_path.is_file())
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["operator_count"], 1)
            self.assertIn("# 数据集静态审计报告", md_path.read_text(encoding="utf-8"))

    def test_audit_extracts_test_tensor_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "datasets"
            op_dir = input_dir / "demo_op"
            op_dir.mkdir(parents=True)
            (op_dir / "demo_op.py").write_text(
                "\n".join(
                    [
                        "import triton",
                        "import triton.language as tl",
                        "@triton.jit",
                        "def kernel(x):",
                        "    return",
                        "def demo_op(x):",
                        "    return x",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (op_dir / "test_demo_op_1.py").write_text(
                "\n".join(
                    [
                        "import torch",
                        "from demo_op import demo_op",
                        "def test_demo_op():",
                        "    device = 'npu'",
                        "    m, k = 32, 16",
                        "    x = torch.randn(m, k, device=device, dtype=torch.float32)",
                        "    y = torch.tensor([0, 16, 32], device=device, dtype=torch.int32)",
                        "    assert demo_op(x) is x",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            report = audit_dataset(input_dir)
            hints = report["operators"][0]["test_hints"]
            tensors = hints["tensor_creations"]

            self.assertEqual(hints["syntax_ok"], True)
            self.assertEqual(tensors[0]["assigned_to"], "x")
            self.assertEqual(tensors[0]["shape"], [32, 16])
            self.assertEqual(tensors[0]["dtype"], "torch.float32")
            self.assertEqual(tensors[0]["device"], "npu")
            self.assertEqual(tensors[1]["assigned_to"], "y")
            self.assertEqual(tensors[1]["shape"], [3])
            self.assertEqual(tensors[1]["dtype"], "torch.int32")

    def test_audit_extracts_shape_observations_from_all_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "datasets"
            op_dir = input_dir / "demo_op"
            op_dir.mkdir(parents=True)
            (op_dir / "demo_op.py").write_text(
                "def demo_op(x):\n    return x\n",
                encoding="utf-8",
            )
            (op_dir / "test_demo_op_1.py").write_text(
                "\n".join(
                    [
                        "import torch",
                        "from demo_op import demo_op",
                        "def test_small():",
                        "    m = 32",
                        "    n = get_runtime_n()",
                        "    x = torch.randn(m, n, dtype=torch.float16).to(torch.float32)",
                        "    y = torch.zeros(m, 16)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (op_dir / "test_demo_op_2.py").write_text(
                "\n".join(
                    [
                        "import torch",
                        "if __name__ == '__main__':",
                        "    module_tensor = torch.empty(0, 8)",
                        "class TestDemo:",
                        "    def test_boundaries(self):",
                        "        empty = torch.empty(0, 64, dtype=torch.float32)",
                        "        invalid = torch.ones(-1, True)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            report = audit_dataset(input_dir)
            operator = report["operators"][0]
            observations = operator["shape_observations"]

            self.assertEqual(report["summary"]["test_file_count"], 2)
            self.assertEqual(report["summary"]["shape_observation_count"], 3)
            self.assertEqual(len(operator["test_files"]), 2)
            self.assertEqual(observations[0]["tensor_shapes"], {"x": [32, None], "y": [32, 16]})
            self.assertEqual(
                observations[0]["tensor_dtypes"],
                {"x": None, "y": None},
            )
            self.assertEqual(
                observations[1]["tensor_shapes"],
                {"module_tensor": [0, 8]},
            )
            self.assertEqual(
                observations[2]["tensor_shapes"],
                {"empty": [0, 64], "invalid": [None, None]},
            )
            self.assertIn("::<module>", observations[1]["case_id"])
            self.assertIn("TestDemo.test_boundaries", observations[2]["source_ref"])
            self.assertEqual(len(observations[0]["signature"]), 64)

    def test_audit_reports_shape_consistency_matrix_and_cli_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "datasets"
            output_dir = tmp_path / "audit"
            op_dir = input_dir / "demo_op"
            op_dir.mkdir(parents=True)
            (op_dir / "demo_op.py").write_text(
                "def demo_op(x, y):\n    return x\n",
                encoding="utf-8",
            )
            (op_dir / "test_demo_op_1.py").write_text(
                "\n".join(
                    [
                        "import torch",
                        "def test_consistent():",
                        "    x = torch.zeros(32)",
                        "    y = torch.zeros(32)",
                        "def test_unknown():",
                        "    n = runtime_n()",
                        "    x = torch.zeros(32)",
                        "    y = torch.zeros(n)",
                        "def test_inconsistent():",
                        "    x = torch.zeros(32)",
                        "    y = torch.zeros(64)",
                        "def test_constraint_violation():",
                        "    x = torch.zeros(8)",
                        "    y = torch.zeros(8)",
                        "def test_max_violation():",
                        "    x = torch.zeros(72)",
                        "    y = torch.zeros(72)",
                        "def test_divisible_violation():",
                        "    x = torch.zeros(34)",
                        "    y = torch.zeros(34)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            contract = ShapeContract(
                op_name="demo_op",
                symbolic_dimensions=[
                    SymbolicDimension(
                        "M",
                        [TensorAxisBinding("x", 0), TensorAxisBinding("y", 0)],
                    )
                ],
                constraints=[
                    ShapeConstraint("min_inclusive", "M", 16),
                    ShapeConstraint("max_inclusive", "M", 64),
                    ShapeConstraint("divisible_by", "M", 8),
                ],
            )

            report = audit_dataset(input_dir, kernel="demo_op", shape_contract=contract)
            matrix = report["operators"][0]["shape_consistency"]

            self.assertEqual(
                matrix["summary"],
                {"total": 6, "consistent": 1, "unknown": 1, "inconsistent": 4},
            )
            self.assertEqual(
                [item["status"] for item in matrix["results"]],
                [
                    "consistent",
                    "unknown",
                    "inconsistent",
                    "inconsistent",
                    "inconsistent",
                    "inconsistent",
                ],
            )
            self.assertEqual(report["summary"]["shape_consistency"], matrix["summary"])
            self.assertEqual(
                [item["status"] for item in matrix["results"][0]["symbols"][0]["constraints"]],
                ["consistent", "consistent", "consistent"],
            )
            self.assertEqual(
                [item["status"] for item in matrix["results"][1]["symbols"][0]["constraints"]],
                ["unknown", "unknown", "unknown"],
            )
            self.assertEqual(
                [issue["code"] for issue in matrix["results"][3]["issues"]],
                ["min_inclusive_violation"],
            )
            self.assertEqual(
                [issue["code"] for issue in matrix["results"][4]["issues"]],
                ["max_inclusive_violation"],
            )
            self.assertEqual(
                [issue["code"] for issue in matrix["results"][5]["issues"]],
                ["divisible_by_violation"],
            )

            contract_path = tmp_path / "shape_contract.json"
            contract_path.write_text(json.dumps(contract.to_dict()), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_dataset.py"),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--kernel",
                    "demo_op",
                    "--shape-contract",
                    str(contract_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            cli_report = json.loads((output_dir / "dataset_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(cli_report["summary"]["shape_consistency"], matrix["summary"])
            self.assertIn("Shape 契约一致性", (output_dir / "dataset_audit.md").read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValueError, "not in the selected operators"):
                audit_dataset(
                    input_dir,
                    kernel="demo_op",
                    shape_contract=ShapeContract("other_op", []),
                )

    def test_shape_contract_rejects_invalid_explicit_constraints(self) -> None:
        dimension = SymbolicDimension("M", [TensorAxisBinding("x", 0)])
        with self.assertRaisesRegex(ValueError, "must be positive"):
            ShapeConstraint("divisible_by", "M", 0)
        with self.assertRaisesRegex(ValueError, "unknown dimensions"):
            ShapeContract(
                "demo_op",
                [dimension],
                constraints=[ShapeConstraint("min_inclusive", "N", 1)],
            )
        with self.assertRaisesRegex(ValueError, "minimum exceeds maximum"):
            ShapeContract(
                "demo_op",
                [dimension],
                constraints=[
                    ShapeConstraint("min_inclusive", "M", 65),
                    ShapeConstraint("max_inclusive", "M", 64),
                ],
            )


if __name__ == "__main__":
    unittest.main()

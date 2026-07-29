"""Static-gate tests for import-alias resolution and taint propagation."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.executors import validate_static_structure


def _checks(code: str):
    return validate_static_structure(code, [])


class TritonSemanticAliasTests(unittest.TestCase):
    def test_module_alias_jit_and_arange_are_still_gated(self) -> None:
        code = "\n".join(
            [
                "import triton as tr",
                "import triton.language as T",
                "",
                "@tr.jit",
                "def kernel(n):",
                "    T.arange(0, n)",
                "",
            ]
        )
        checks = _checks(code)
        self.assertTrue(checks["has_triton_jit"])
        self.assertFalse(checks["triton_semantics_ok"])
        self.assertEqual(checks["triton_semantic_errors"][0]["code"], "dynamic_tl_arange")

    def test_from_import_jit_and_arange_are_still_gated(self) -> None:
        code = "\n".join(
            [
                "from triton import jit",
                "from triton.language import arange",
                "",
                "@jit",
                "def kernel(n):",
                "    arange(0, n)",
                "",
            ]
        )
        checks = _checks(code)
        self.assertTrue(checks["has_triton_jit"])
        self.assertFalse(checks["triton_semantics_ok"])

    def test_aliased_constexpr_annotation_is_not_runtime(self) -> None:
        code = "\n".join(
            [
                "import triton",
                "import triton.language as T",
                "",
                "@triton.jit",
                "def kernel(BLOCK: T.constexpr):",
                "    T.arange(0, BLOCK)",
                "",
            ]
        )
        self.assertTrue(_checks(code)["triton_semantics_ok"])

    def test_unrelated_module_aliased_as_tl_is_not_flagged(self) -> None:
        code = "\n".join(
            [
                "import triton",
                "import mypkg as tl",
                "",
                "@triton.jit",
                "def kernel(n):",
                "    tl.arange(0, n)",
                "",
            ]
        )
        self.assertTrue(_checks(code)["triton_semantics_ok"])

    def test_aug_assign_propagates_runtime_taint(self) -> None:
        code = "\n".join(
            [
                "import triton",
                "import triton.language as tl",
                "",
                "@triton.jit",
                "def kernel(n):",
                "    m = 0",
                "    m += n",
                "    tl.arange(0, m)",
                "",
            ]
        )
        checks = _checks(code)
        self.assertFalse(checks["triton_semantics_ok"])
        self.assertIn("m", checks["triton_semantic_errors"][0]["runtime_symbols"])

    def test_aliased_runtime_call_taints_assignment(self) -> None:
        code = "\n".join(
            [
                "import triton",
                "import triton.language as T",
                "",
                "@triton.jit",
                "def kernel(BLOCK: T.constexpr):",
                "    n = T.program_id(0)",
                "    T.arange(0, n)",
                "",
            ]
        )
        self.assertFalse(_checks(code)["triton_semantics_ok"])


class ImportGateTests(unittest.TestCase):
    def test_plain_triton_language_import_passes_without_tl_injection(self) -> None:
        code = "import triton.language\n"
        checks = _checks(code)
        self.assertTrue(checks["imports_ok"])
        self.assertNotIn("tl", checks["imports"])

    def test_unrelated_from_import_of_tl_is_rejected(self) -> None:
        code = "import triton\nfrom mypkg import tl\n"
        self.assertFalse(_checks(code)["imports_ok"])

    def test_vllm_reexport_still_passes(self) -> None:
        code = "from vllm.triton_utils import tl, triton\n"
        self.assertTrue(_checks(code)["imports_ok"])

    def test_aliased_imports_pass(self) -> None:
        code = "import triton as tr\nimport triton.language as T\n"
        self.assertTrue(_checks(code)["imports_ok"])


if __name__ == "__main__":
    unittest.main()

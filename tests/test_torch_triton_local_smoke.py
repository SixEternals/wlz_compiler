"""D2-local CUDA/Triton smoke; skipped in environments without its optional deps."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

try:
    import torch
    import triton  # noqa: F401
except ImportError:  # The repository's base Python intentionally has no Torch.
    torch = None

_CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()

from wlz_optimizer.case_catalog import materialize_rms_norm_public_case
from wlz_optimizer.input_materializer import clone_inputs_for_run, materialize_inputs
from wlz_optimizer.torch_backend import TorchTensorBackend


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "work/official_triton_agent/datasets"
RMS_MODULE_ROOT = DATASET_ROOT / "_rms_norm_kernel"
SELECTIVE_MODULE_ROOT = DATASET_ROOT / "_selective_scan_update_kernel"


@unittest.skipUnless(_CUDA_AVAILABLE, "D2-local requires Torch, Triton, and a CUDA device")
class TorchTritonLocalSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(RMS_MODULE_ROOT))
        sys.path.insert(0, str(SELECTIVE_MODULE_ROOT))
        from _rms_norm_kernel import rms_norm
        from test__selective_scan_update_kernel_1 import SelectiveScanUpdateReference

        cls.rms_norm = staticmethod(rms_norm)
        candidate_path = os.environ.get("WLZ_SELECTIVE_CANDIDATE")
        cls._candidate_module_name = None
        if candidate_path:
            candidate_file = Path(candidate_path).resolve()
            if not candidate_file.is_file():
                raise FileNotFoundError(f"selective candidate not found: {candidate_file}")
            module_name = "_wlz_generated_selective_scan_candidate"
            spec = importlib.util.spec_from_file_location(module_name, candidate_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load selective candidate: {candidate_file}")
            candidate_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = candidate_module
            spec.loader.exec_module(candidate_module)
            cls._candidate_module_name = module_name
            cls.selective_state_update = staticmethod(
                candidate_module.selective_state_update
            )
        else:
            from _selective_scan_update_kernel import selective_state_update

            cls.selective_state_update = staticmethod(selective_state_update)
        cls.selective_reference = SelectiveScanUpdateReference()
        cls.backend = TorchTensorBackend("cuda")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._candidate_module_name is not None:
            sys.modules.pop(cls._candidate_module_name, None)
        for path in (str(SELECTIVE_MODULE_ROOT), str(RMS_MODULE_ROOT)):
            if path in sys.path:
                sys.path.remove(path)

    def test_official_rms_baseline_and_wrong_candidate(self) -> None:
        case = materialize_rms_norm_public_case(DATASET_ROOT)
        pristine = materialize_inputs(case, self.backend)
        reference_inputs = clone_inputs_for_run(pristine, self.backend)
        candidate_inputs = clone_inputs_for_run(pristine, self.backend)
        eps = case.inputs.scalars["eps"]

        reference_x = reference_inputs.tensors["input_tensor"]
        reference_weight = reference_inputs.tensors["weight_tensor"]
        variance = reference_x.square().mean(dim=-1, keepdim=True)
        reference = reference_x * torch.rsqrt(variance + eps) * reference_weight

        candidate = self.rms_norm(
            candidate_inputs.tensors["input_tensor"],
            candidate_inputs.tensors["weight_tensor"],
            eps,
        )
        wrong_candidate = (
            candidate_inputs.tensors["input_tensor"]
            * candidate_inputs.tensors["weight_tensor"]
        )
        self.backend.synchronize()

        maximum_error = (candidate - reference).abs().max().item()
        self.assertTrue(
            torch.allclose(
                candidate,
                reference,
                rtol=case.oracle_policy.rtol,
                atol=case.oracle_policy.atol,
            ),
            f"official RMS Triton baseline max_abs_error={maximum_error}",
        )
        self.assertFalse(
            torch.allclose(
                wrong_candidate,
                reference,
                rtol=case.oracle_policy.rtol,
                atol=case.oracle_policy.atol,
            )
        )
        print(
            "D2_LOCAL_RMS",
            {
                "shape": tuple(candidate.shape),
                "max_abs_error": maximum_error,
                "baseline_passed": True,
                "wrong_candidate_rejected": True,
            },
        )

    def test_official_selective_scan_stateful_fresh_rerun(self) -> None:
        cases = [
            {"name": "dstate16_all_features", "dstate": 16, "features": True},
            {"name": "dstate17_minimal", "dstate": 17},
            {"name": "dstate33_tied", "dstate": 33, "tied": True},
            {
                "name": "dstate65_remapped",
                "dstate": 65,
                "features": True,
                "state_batch_indices": [1, 2, 0],
            },
            {"name": "dstate129_minimal", "dstate": 129},
        ]
        metrics = []
        for seed, case in enumerate(cases):
            with self.subTest(case=case["name"]):
                metrics.append(self._run_selective_case(seed, case))
        print("D2_LOCAL_SELECTIVE_SCAN_MATRIX", metrics)

    def _run_selective_case(self, seed, case):
        torch.manual_seed(seed)
        batch = len(case.get("state_batch_indices", [])) or 2
        nheads, dim, dstate, ngroups = 4, 35, case["dstate"], 2
        state = torch.randn(batch, nheads, dim, dstate, device="cuda")
        pristine_state = state.clone()
        x = torch.randn(batch, nheads, dim, device="cuda")
        dt = torch.randn_like(x)
        A = torch.randn(nheads, dim, dstate, device="cuda")
        B = torch.randn(batch, ngroups, dstate, device="cuda")
        C = torch.randn_like(B)
        features = case.get("features", False)
        D = torch.randn(nheads, dim, device="cuda") if features else None
        z = torch.randn_like(x) if features else None
        dt_bias = torch.randn(nheads, dim, device="cuda") if features else None
        if case.get("tied"):
            dt = torch.randn(batch, nheads, 1, device="cuda").expand_as(x)
            A = torch.randn(nheads, 1, 1, device="cuda").expand_as(A)
            dt_bias = torch.randn(nheads, 1, device="cuda").expand(nheads, dim)
        indices = case.get("state_batch_indices")
        state_batch_indices = (
            torch.tensor(indices, device="cuda", dtype=torch.int32)
            if indices is not None
            else None
        )

        reference_out, reference_state = self.selective_reference(
            state.clone(),
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=features,
            state_batch_indices=state_batch_indices,
        )

        def run_fresh():
            fresh_state = state.clone()
            out = torch.zeros_like(x)
            self.selective_state_update(
                fresh_state,
                x,
                dt,
                A,
                B,
                C,
                D=D,
                z=z,
                dt_bias=dt_bias,
                dt_softplus=features,
                state_batch_indices=state_batch_indices,
                out=out,
            )
            return out, fresh_state

        first_out, first_state = run_fresh()
        second_out, second_state = run_fresh()
        self.backend.synchronize()

        output_error = (first_out - reference_out).abs().max().item()
        state_error = (first_state - reference_state).abs().max().item()
        self.assertTrue(
            torch.allclose(first_out, reference_out, rtol=1e-4, atol=1e-4),
            f"{case['name']} output max_abs_error={output_error}",
        )
        self.assertTrue(
            torch.allclose(first_state, reference_state, rtol=1e-4, atol=1e-4),
            f"{case['name']} state max_abs_error={state_error}",
        )
        self.assertTrue(torch.equal(first_out, second_out))
        self.assertTrue(torch.equal(first_state, second_state))
        self.assertTrue(torch.equal(state, pristine_state))
        self.assertFalse(torch.equal(state, first_state))
        self.assertNotEqual(
            first_state.untyped_storage().data_ptr(),
            second_state.untyped_storage().data_ptr(),
        )
        return {
            "case": case["name"],
            "dstate": dstate,
            "output_max_abs_error": output_error,
            "state_max_abs_error": state_error,
            "reruns_equal": True,
            "pristine_unchanged": True,
            "storage_isolated": True,
        }


if __name__ == "__main__":
    unittest.main()

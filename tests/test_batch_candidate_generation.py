import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from wlz_optimizer.budget import BudgetController, BudgetLimits
from wlz_optimizer.hash_utils import sha256_text

ROOT = Path(__file__).resolve().parents[1]


def load_batch_module():
    path = ROOT / "scripts" / "generate_official_candidates_batch.py"
    spec = importlib.util.spec_from_file_location("batch_candidate_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_generator_module():
    path = ROOT / "scripts" / "generate_official_candidate.py"
    spec = importlib.util.spec_from_file_location("mutation_parent_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchCandidateGenerationTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        datasets = root / "datasets"
        for operator in ("op_a", "op_b"):
            operator_dir = datasets / operator
            operator_dir.mkdir(parents=True)
            baseline = f"def {operator}(x):\n    return x\n"
            variant = f"def {operator}(x):\n    return x + 1\n"
            (operator_dir / f"{operator}.py").write_text(baseline, encoding="utf-8")
            (operator_dir / f"{operator}_1.py").write_text(variant, encoding="utf-8")
        return datasets

    def _write_pass(
        self,
        output: Path,
        operator: str,
        candidate_id: str,
        code: str | None = None,
        random_seed: int | None = None,
        parent_path: Path | None = None,
    ) -> None:
        candidate_dir = output / operator
        candidate_dir.mkdir(parents=True, exist_ok=True)
        code = code or f"def {operator}(x):\n    return x + 2\n"
        (candidate_dir / f"{candidate_id}.py").write_text(code, encoding="utf-8")
        manifest = {
            "candidate": {
                "id": candidate_id,
                "op_name": operator,
                "status": "static_pass",
                "code_hash": sha256_text(code),
                "generation": 0,
                "parent_ids": [],
            },
            "static_evaluation": {"passed": True},
            "import_evaluation": {
                "status": "imported",
                "phase": "module_import",
                "error_type": None,
                "error_message": None,
            },
            "rejection_error": None,
        }
        if random_seed is not None and parent_path is not None:
            parent_hash = sha256_text(parent_path.read_text(encoding="utf-8"))
            parent_id = f"seed-{parent_hash[:12]}"
            parent_generation = 0
            parent_manifest = parent_path.with_suffix(".manifest.json")
            if parent_manifest.is_file():
                parent_candidate = json.loads(
                    parent_manifest.read_text(encoding="utf-8")
                )["candidate"]
                parent_id = parent_candidate["id"]
                parent_generation = parent_candidate["generation"]
            manifest["candidate"]["parent_ids"] = [parent_id]
            manifest["candidate"]["generation"] = parent_generation + 1
            manifest["random_seed"] = random_seed
            manifest["parent_path"] = str(parent_path.resolve())
            manifest["parent_sha256"] = parent_hash
        (candidate_dir / f"{candidate_id}.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_dry_run_skips_valid_candidate_without_generation(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            self._write_pass(output, "op_a", "existing-a")
            calls = []
            report = module.run_batch(
                root,
                datasets,
                output,
                random_seed=5,
                max_new_calls=0,
                dry_run=True,
                generate_fn=lambda *args: calls.append(args),
                contract_module=SimpleNamespace(interface_contract_error=lambda *_: None),
            )
        self.assertEqual(calls, [])
        self.assertEqual(report["status_counts"], {"skipped_static_pass": 1, "planned": 1})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            self._write_pass(output, "op_a", "existing-a")
            report = module.run_batch(
                root,
                datasets,
                output,
                random_seed=5,
                max_new_calls=0,
                generate_fn=lambda *args: calls.append(args),
                contract_module=SimpleNamespace(interface_contract_error=lambda *_: None),
            )
        self.assertEqual(calls, [])
        self.assertEqual(
            report["status_counts"],
            {"skipped_static_pass": 1, "deferred_call_budget": 1},
        )

    def test_generated_candidate_is_skipped_on_resume(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            self._write_pass(output, "op_a", "existing-a")
            calls = []

            def fake_generate(work, data, operator, parent, out, seed):
                calls.append((operator, parent.name, seed))
                self._write_pass(out, operator, "generated-b")
                return {"candidate": {"id": "generated-b"}}

            contract = SimpleNamespace(interface_contract_error=lambda *_: None)
            first = module.run_batch(
                root, datasets, output, 5, 1,
                generate_fn=fake_generate, contract_module=contract,
            )
            second = module.run_batch(
                root, datasets, output, 5, 1,
                generate_fn=fake_generate, contract_module=contract,
            )
        self.assertEqual(calls, [("op_b", "op_b_1.py", 5)])
        self.assertEqual(first["status_counts"], {"skipped_static_pass": 1, "generated_static_pass": 1})
        self.assertEqual(second["status_counts"], {"skipped_static_pass": 2})

    def test_mutation_loop_advances_parent_only_after_valid_child(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            calls = []
            correctness_calls = []

            def fake_generate(work, data, operator, parent, out, seed):
                ordinal = len(calls)
                calls.append((parent.name, seed))
                if ordinal == 1:
                    raise RuntimeError("synthetic mutation rejection")
                candidate_id = f"child-{ordinal}"
                code = f"def {operator}(x):\n    return x + {ordinal + 2}\n"
                self._write_pass(
                    out, operator, candidate_id, code, random_seed=seed, parent_path=parent
                )
                return {"candidate": {"id": candidate_id}}

            def correctness_gate(manifest_path):
                candidate_id = manifest_path.name.removesuffix(".manifest.json")
                correctness_calls.append(candidate_id)
                status = {
                    "child-0": "passed",
                    "child-2": "unknown",
                    "child-3": "failed",
                }[candidate_id]
                return {
                    "status": status,
                    "eligible_for_performance": status == "passed",
                    "blocking_reasons": [] if status == "passed" else [f"{status}_case"],
                    "results": [{"oracle_status": status}],
                    "decision": {"eligible_for_performance": status == "passed"},
                }

            report = module.run_mutation_loop(
                root,
                datasets,
                output,
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=4,
                generate_fn=fake_generate,
                correctness_gate=correctness_gate,
                beam_width=1,
            )
            manifests = {
                candidate_id: json.loads(
                    (output / f"op_a/{candidate_id}.manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                for candidate_id in correctness_calls
            }

        self.assertEqual(
            calls,
            [
                ("op_a.py", 5),
                ("child-0.py", 6),
                ("child-0.py", 7),
                ("child-0.py", 8),
            ],
        )
        self.assertEqual(
            [record["status"] for record in report["records"]],
            [
                "correctness_pass",
                "generation_failed",
                "correctness_unknown",
                "correctness_failed",
            ],
        )
        self.assertEqual(
            report["status_counts"],
            {
                "correctness_pass": 1,
                "generation_failed": 1,
                "correctness_unknown": 1,
                "correctness_failed": 1,
            },
        )
        self.assertTrue(
            report["final_admitted_parent_path"].endswith("generated/op_a/child-0.py")
        )
        self.assertEqual(correctness_calls, ["child-0", "child-2", "child-3"])
        self.assertEqual(manifests["child-0"]["candidate"]["status"], "static_pass")
        for candidate_id in ("child-2", "child-3"):
            self.assertEqual(manifests[candidate_id]["candidate"]["status"], "rejected")
            self.assertIn("correctness_evaluation", manifests[candidate_id])
            self.assertIsNotNone(manifests[candidate_id]["rejection_error"])
            self.assertTrue(manifests[candidate_id]["static_evaluation"]["passed"])
            self.assertEqual(
                manifests[candidate_id]["import_evaluation"]["status"], "imported"
            )

    def test_budgeted_mutation_loop_records_and_restores_usage(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            checkpoint = root / "mutation-loop.json"
            budget = BudgetController(BudgetLimits(1_000, 300))
            injected = []

            def fake_generate(
                work, data, operator, parent, out, seed, *, budget_controller
            ):
                injected.append(budget_controller)
                reservation = budget_controller.reserve(10, 20, 0, 1).reservation
                budget_controller.commit(reservation, 17)
                candidate_id = f"child-{len(injected) - 1}"
                code = f"def {operator}(x):\n    return x + {len(injected) + 1}\n"
                self._write_pass(
                    out,
                    operator,
                    candidate_id,
                    code,
                    random_seed=seed,
                    parent_path=parent,
                )
                manifest_path = out / operator / f"{candidate_id}.manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["candidate"].update(
                    {"model_used": "deepseek-v4-pro", "prompt_id": "a" * 64}
                )
                manifest["llm_stats"] = {
                    "calls": [
                        {
                            "model": "deepseek-v4-pro",
                            "prompt_sha256": "a" * 64,
                            "request_fingerprint": "b" * 64,
                            "status": "succeeded",
                            "usage": {"total_tokens": 17},
                        }
                    ]
                }
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                return manifest

            report = module.run_mutation_loop(
                root,
                datasets,
                output,
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=2,
                generate_fn=fake_generate,
                checkpoint_path=checkpoint,
                budget_controller=budget,
            )
            restored_budget = BudgetController(BudgetLimits(1_000, 300))
            resumed = module.run_mutation_loop(
                root,
                datasets,
                output,
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=0,
                checkpoint_path=checkpoint,
                resume=True,
                budget_controller=restored_budget,
            )
            tampered = json.loads(checkpoint.read_text(encoding="utf-8"))
            tampered["budget_checkpoint"]["used_tokens"] = 0
            tampered["budget_checkpoint"]["remaining_tokens"] = 1_000
            checkpoint.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "budget state does not match attempt history"
            ):
                module.run_mutation_loop(
                    root,
                    datasets,
                    output,
                    "op_a",
                    datasets / "op_a/op_a.py",
                    random_seed=5,
                    max_new_calls=0,
                    checkpoint_path=checkpoint,
                    resume=True,
                    budget_controller=BudgetController(BudgetLimits(1_000, 300)),
                )

        self.assertEqual(injected, [budget, budget])
        self.assertEqual(report["budget_checkpoint"]["used_tokens"], 34)
        self.assertEqual(
            [record["budget_before"]["used_tokens"] for record in report["records"]],
            [0, 17],
        )
        self.assertEqual(
            [record["budget_after"]["used_tokens"] for record in report["records"]],
            [17, 34],
        )
        self.assertEqual(report["records"][0]["llm"]["model"], "deepseek-v4-pro")
        self.assertEqual(report["records"][0]["llm"]["prompt_sha256"], "a" * 64)
        self.assertEqual(
            report["records"][0]["llm"]["usage"]["total_tokens"], 17
        )
        self.assertEqual(restored_budget.snapshot().used_tokens, 34)
        self.assertEqual(resumed["budget_checkpoint"]["used_tokens"], 34)

    def test_failed_mutation_keeps_safe_llm_provenance(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)

            def fake_generate(*args, **kwargs):
                budget = kwargs["budget_controller"]
                reservation = budget.reserve(10, 20, 0, 1).reservation
                budget.mark_uncertain(reservation)
                error = RuntimeError("synthetic model failure")
                error._wlz_llm_stats = {
                    "calls": [
                        {
                            "model": "deepseek-v4-pro",
                            "prompt_sha256": "c" * 64,
                            "request_fingerprint": "d" * 64,
                            "status": "failed",
                        }
                    ]
                }
                raise error

            report = module.run_mutation_loop(
                root,
                datasets,
                root / "generated",
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=3,
                generate_fn=fake_generate,
                budget_controller=BudgetController(BudgetLimits(1_000, 300)),
            )

        record = report["records"][0]
        self.assertEqual(len(report["records"]), 1)
        self.assertEqual(report["calls_started_this_run"], 1)
        self.assertEqual(
            report["budget_checkpoint"]["stop_reason"], "unknown_in_flight_call"
        )
        self.assertEqual(record["status"], "generation_failed")
        self.assertEqual(record["llm"]["model"], "deepseek-v4-pro")
        self.assertEqual(record["llm"]["prompt_sha256"], "c" * 64)
        self.assertEqual(record["llm"]["status"], "failed")
        self.assertEqual(record["new_artifacts"], [])

    def test_budget_denial_stops_remaining_logical_attempts(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            checkpoint = root / "mutation-loop.json"
            calls = []

            def fake_generate(*args, **kwargs):
                calls.append(1)
                decision = kwargs["budget_controller"].reserve(10, 20, 0, 1)
                self.assertFalse(decision.allowed)
                error = RuntimeError("LLM budget denied request: token_limit")
                error._wlz_llm_stats = {
                    "calls": [
                        {
                            "model": "deepseek-v4-pro",
                            "prompt_sha256": "f" * 64,
                            "status": "failed",
                            "error_type": "budget_denied",
                        }
                    ]
                }
                raise error

            report = module.run_mutation_loop(
                root,
                datasets,
                root / "generated",
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=3,
                generate_fn=fake_generate,
                checkpoint_path=checkpoint,
                budget_controller=BudgetController(BudgetLimits(10, 300)),
            )
            resumed = module.run_mutation_loop(
                root,
                datasets,
                root / "generated",
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=3,
                generate_fn=fake_generate,
                checkpoint_path=checkpoint,
                resume=True,
                budget_controller=BudgetController(BudgetLimits(10, 300)),
            )

        self.assertEqual(calls, [1])
        self.assertEqual(len(report["records"]), 1)
        self.assertEqual(report["calls_started_this_run"], 1)
        self.assertEqual(report["records"][0]["llm"]["error_type"], "budget_denied")
        self.assertEqual(report["budget_denial_reason"], "token_limit")
        self.assertEqual(resumed["calls_started_this_run"], 0)
        self.assertEqual(resumed["budget_denial_reason"], "token_limit")

    def test_budgeted_resume_rejects_removed_denial_reason(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            checkpoint = root / "mutation-loop.json"
            calls = []

            def denied_generate(*args, **kwargs):
                calls.append(1)
                decision = kwargs["budget_controller"].reserve(10, 20, 0, 1)
                self.assertFalse(decision.allowed)
                error = RuntimeError("LLM budget denied request: token_limit")
                error._wlz_llm_stats = {
                    "calls": [{"status": "failed", "error_type": "budget_denied"}]
                }
                raise error

            module.run_mutation_loop(
                root,
                datasets,
                output,
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=1,
                generate_fn=denied_generate,
                checkpoint_path=checkpoint,
                budget_controller=BudgetController(BudgetLimits(10, 300)),
            )
            tampered = json.loads(checkpoint.read_text(encoding="utf-8"))
            tampered["budget_denial_reason"] = None
            checkpoint.write_text(json.dumps(tampered), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid mutation checkpoint"):
                module.run_mutation_loop(
                    root,
                    datasets,
                    output,
                    "op_a",
                    datasets / "op_a/op_a.py",
                    random_seed=5,
                    max_new_calls=1,
                    generate_fn=denied_generate,
                    checkpoint_path=checkpoint,
                    resume=True,
                    budget_controller=BudgetController(BudgetLimits(10, 300)),
                )

        self.assertEqual(calls, [1])

    def test_budgeted_resume_rejects_removed_unknown_stop_reason(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            checkpoint = root / "mutation-loop.json"
            calls = []

            def uncertain_generate(*args, **kwargs):
                calls.append(1)
                budget = kwargs["budget_controller"]
                reservation = budget.reserve(10, 20, 0, 1).reservation
                budget.mark_uncertain(reservation)
                raise RuntimeError("synthetic uncertain call")

            module.run_mutation_loop(
                root,
                datasets,
                output,
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=1,
                generate_fn=uncertain_generate,
                checkpoint_path=checkpoint,
                budget_controller=BudgetController(BudgetLimits(1_000, 300)),
            )
            tampered = json.loads(checkpoint.read_text(encoding="utf-8"))
            tampered["budget_checkpoint"]["stop_reason"] = None
            checkpoint.write_text(json.dumps(tampered), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid mutation checkpoint"):
                module.run_mutation_loop(
                    root,
                    datasets,
                    output,
                    "op_a",
                    datasets / "op_a/op_a.py",
                    random_seed=5,
                    max_new_calls=1,
                    generate_fn=uncertain_generate,
                    checkpoint_path=checkpoint,
                    resume=True,
                    budget_controller=BudgetController(BudgetLimits(1_000, 300)),
                )

        self.assertEqual(calls, [1])

    def test_budgeted_resume_accepts_wall_expiry_during_final_checkpoint(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            checkpoint = root / "mutation-loop.json"
            calls = []
            ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2])
            last_tick = [0.0]

            def clock():
                last_tick[0] = next(ticks, last_tick[0])
                return last_tick[0]

            def failed_generate(*args, **kwargs):
                calls.append(1)
                raise RuntimeError("synthetic generation failure")

            report = module.run_mutation_loop(
                root,
                datasets,
                output,
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=1,
                generate_fn=failed_generate,
                checkpoint_path=checkpoint,
                budget_controller=BudgetController(
                    BudgetLimits(1_000, 1.0), clock=clock
                ),
            )
            resumed = module.run_mutation_loop(
                root,
                datasets,
                output,
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=1,
                generate_fn=failed_generate,
                checkpoint_path=checkpoint,
                resume=True,
                budget_controller=BudgetController(
                    BudgetLimits(1_000, 1.0), clock=lambda: 10.0
                ),
            )

        self.assertIsNone(report["records"][0]["budget_after"]["stop_reason"])
        self.assertEqual(
            report["budget_checkpoint"]["stop_reason"], "wall_time_limit"
        )
        self.assertEqual(resumed["calls_started_this_run"], 0)
        self.assertEqual(calls, [1])

    def test_budgeted_resume_rejects_unknown_in_flight_attempt(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            checkpoint = root / "mutation-loop.json"
            with self.assertRaises(KeyboardInterrupt):
                module.run_mutation_loop(
                    root,
                    datasets,
                    root / "generated",
                    "op_a",
                    datasets / "op_a/op_a.py",
                    random_seed=5,
                    max_new_calls=1,
                    generate_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
                        KeyboardInterrupt()
                    ),
                    checkpoint_path=checkpoint,
                    budget_controller=BudgetController(BudgetLimits(1_000, 300)),
                )
            with self.assertRaisesRegex(
                ValueError, "budgeted checkpoint has an unresolved in-flight attempt"
            ):
                module.run_mutation_loop(
                    root,
                    datasets,
                    root / "generated",
                    "op_a",
                    datasets / "op_a/op_a.py",
                    random_seed=5,
                    max_new_calls=1,
                    checkpoint_path=checkpoint,
                    resume=True,
                    retry_in_flight=True,
                    budget_controller=BudgetController(BudgetLimits(1_000, 300)),
                )

    def test_width_two_frontier_round_robins_and_rejects_non_admitted(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            checkpoint = root / "mutation-loop.json"
            calls = []

            def fake_generate(work, data, operator, parent, out, seed):
                ordinal = len(calls)
                calls.append((parent.name, seed))
                candidate_id = f"child-{ordinal}"
                code_ordinal = 0 if ordinal == 2 else ordinal
                code = f"def {operator}(x):\n    return x + {code_ordinal + 2}\n"
                self._write_pass(
                    out, operator, candidate_id, code, random_seed=seed, parent_path=parent
                )
                manifest_path = out / operator / f"{candidate_id}.manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if ordinal == 3:
                    manifest["static_evaluation"]["passed"] = False
                elif ordinal == 4:
                    manifest["import_evaluation"]["status"] = "import_error"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                return {"candidate": {"id": candidate_id}}

            def correctness_gate(manifest_path):
                if manifest_path.name == "child-5.manifest.json":
                    raise RuntimeError("synthetic gate failure")
                return {
                    "status": "passed",
                    "eligible_for_performance": True,
                    "blocking_reasons": [],
                    "results": [{"oracle_status": "passed"}],
                }

            first_report = module.run_mutation_loop(
                root,
                datasets,
                output,
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=2,
                generate_fn=fake_generate,
                correctness_gate=correctness_gate,
                beam_width=2,
                checkpoint_path=checkpoint,
            )
            report = module.run_mutation_loop(
                root,
                datasets,
                output,
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=4,
                generate_fn=fake_generate,
                correctness_gate=correctness_gate,
                beam_width=2,
                checkpoint_path=checkpoint,
                resume=True,
            )
            duplicate_manifest = json.loads(
                (output / "op_a/child-2.manifest.json").read_text(encoding="utf-8")
            )
            gate_error_manifest = json.loads(
                (output / "op_a/child-5.manifest.json").read_text(encoding="utf-8")
            )
            original_checkpoint = checkpoint.read_text(encoding="utf-8")

            def rejects_checkpoint(mutator):
                state = json.loads(original_checkpoint)
                mutator(state)
                checkpoint.write_text(json.dumps(state), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid mutation checkpoint"):
                    module.run_mutation_loop(
                        root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 0,
                        correctness_gate=correctness_gate, beam_width=2,
                        checkpoint_path=checkpoint, resume=True,
                    )
                checkpoint.write_text(original_checkpoint, encoding="utf-8")

            rejects_checkpoint(lambda state: state.__setitem__("operator", "wrong_op"))
            rejects_checkpoint(lambda state: state["frontier"].reverse())
            rejects_checkpoint(lambda state: state.__setitem__("next_parent_index", 1))
            rejects_checkpoint(lambda state: state["seen_code_hashes"].pop())
            rejects_checkpoint(
                lambda state: state["records"][0].__setitem__("frontier_action", "none")
            )
            child_path = output / "op_a/child-1.py"
            child_manifest_path = output / "op_a/child-1.manifest.json"
            original_manifest = child_manifest_path.read_text(encoding="utf-8")
            mismatched = json.loads(original_manifest)
            mismatched["candidate"]["id"] = "child-0"
            child_manifest_path.write_text(json.dumps(mismatched), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest identity mismatch"):
                module.run_mutation_loop(
                    root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 0,
                    correctness_gate=correctness_gate, beam_width=2,
                    checkpoint_path=checkpoint, resume=True,
                )
            child_manifest_path.write_text(original_manifest, encoding="utf-8")
            for label, mutate_manifest in (
                ("seed", lambda item: item.__setitem__("random_seed", 999)),
                ("parent_hash", lambda item: item.__setitem__("parent_sha256", "0" * 64)),
                (
                    "parent_ids",
                    lambda item: item["candidate"].__setitem__("parent_ids", ["fake"]),
                ),
                (
                    "generation",
                    lambda item: item["candidate"].__setitem__("generation", 99),
                ),
            ):
                with self.subTest(manifest_tamper=label):
                    mismatched = json.loads(original_manifest)
                    mutate_manifest(mismatched)
                    child_manifest_path.write_text(
                        json.dumps(mismatched), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "mutation checkpoint"):
                        module.run_mutation_loop(
                            root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 0,
                            correctness_gate=correctness_gate, beam_width=2,
                            checkpoint_path=checkpoint, resume=True,
                        )
            child_manifest_path.write_text(original_manifest, encoding="utf-8")
            original_child = child_path.read_text(encoding="utf-8")
            child_path.write_text("def op_a(x): return -1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                module.run_mutation_loop(
                    root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 0,
                    correctness_gate=correctness_gate, beam_width=2,
                    checkpoint_path=checkpoint, resume=True,
                )
            child_path.write_text(original_child, encoding="utf-8")
            with self.assertRaises(KeyboardInterrupt):
                module.run_mutation_loop(
                    root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 1,
                    generate_fn=lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
                    correctness_gate=correctness_gate, beam_width=2,
                    checkpoint_path=checkpoint, resume=True,
                )
            interrupted = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(interrupted["next_ordinal"], 6)
            self.assertEqual(interrupted["in_flight"]["random_seed"], 11)
            self.assertEqual(
                interrupted["in_flight"]["physical_attempts_maybe_started"], 1
            )
            with self.assertRaisesRegex(ValueError, "unresolved in-flight"):
                module.run_mutation_loop(
                    root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 0,
                    correctness_gate=correctness_gate, beam_width=2,
                    checkpoint_path=checkpoint, resume=True,
                )
            with self.assertRaises(KeyboardInterrupt):
                module.run_mutation_loop(
                    root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 1,
                    generate_fn=lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
                    correctness_gate=correctness_gate, beam_width=2,
                    checkpoint_path=checkpoint, resume=True, retry_in_flight=True,
                )
            retry_interrupted = checkpoint.read_text(encoding="utf-8")
            retry_state = json.loads(retry_interrupted)
            self.assertEqual(
                retry_state["in_flight"]["physical_attempts_maybe_started"], 2
            )
            self.assertTrue(retry_state["in_flight"]["explicit_retry"])
            tampered_retry = json.loads(retry_interrupted)
            tampered_retry["in_flight"]["explicit_retry"] = False
            checkpoint.write_text(json.dumps(tampered_retry), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "in-flight attempt is inconsistent"):
                module.run_mutation_loop(
                    root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 1,
                    correctness_gate=correctness_gate, beam_width=2,
                    checkpoint_path=checkpoint, resume=True, retry_in_flight=True,
                )
            checkpoint.write_text(retry_interrupted, encoding="utf-8")

            def retry_generate(work, data, operator, parent, out, seed):
                ordinal = len(calls)
                calls.append((parent.name, seed))
                candidate_id = f"child-{ordinal}"
                code = f"def {operator}(x):\n    return x + {ordinal + 2}\n"
                self._write_pass(
                    out, operator, candidate_id, code, random_seed=seed, parent_path=parent
                )
                return {"candidate": {"id": candidate_id}}

            retry_report = module.run_mutation_loop(
                root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 1,
                generate_fn=retry_generate, correctness_gate=correctness_gate, beam_width=2,
                checkpoint_path=checkpoint, resume=True, retry_in_flight=True,
            )
            self.assertEqual(retry_report["next_ordinal"], 7)
            self.assertEqual(retry_report["calls_started_this_run"], 1)
            self.assertEqual(retry_report["calls_started"], 9)
            self.assertEqual(retry_report["records"][-1]["random_seed"], 11)
            self.assertTrue(retry_report["records"][-1]["retried_in_flight"])
            self.assertEqual(
                retry_report["records"][-1]["physical_attempts_maybe_started"], 3
            )
            self.assertEqual(
                retry_report["records"][-1]["prior_token_usage"],
                "unknown_may_have_been_consumed",
            )
            with self.assertRaisesRegex(ValueError, "requires an unresolved attempt"):
                module.run_mutation_loop(
                    root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 1,
                    correctness_gate=correctness_gate, beam_width=2,
                    checkpoint_path=checkpoint, resume=True, retry_in_flight=True,
                )

        self.assertEqual(
            calls,
            [
                ("op_a.py", 5),
                ("child-0.py", 6),
                ("op_a.py", 7),
                ("child-1.py", 8),
                ("op_a.py", 9),
                ("child-1.py", 10),
                ("op_a.py", 11),
            ],
        )
        self.assertEqual(first_report["next_ordinal"], 2)
        self.assertEqual(report["next_ordinal"], 6)
        self.assertEqual(report["calls_started_this_run"], 4)
        self.assertEqual(
            [record["status"] for record in report["records"]],
            [
                "correctness_pass",
                "correctness_pass",
                "duplicate_code_hash",
                "generation_failed",
                "generation_failed",
                "generation_failed",
            ],
        )
        self.assertEqual(
            [record["frontier_action"] for record in report["records"]],
            ["append", "replace", "none", "none", "none", "none"],
        )
        self.assertEqual(report["beam_width"], 2)
        self.assertTrue(report["final_admitted_parent_path"].endswith("child-1.py"))
        self.assertEqual(
            [Path(path).name for path in report["final_admitted_parent_paths"]],
            ["op_a.py", "child-1.py"],
        )
        self.assertEqual(duplicate_manifest["candidate"]["status"], "rejected")
        self.assertEqual(
            duplicate_manifest["rejection_error"], "duplicate admitted code hash"
        )
        self.assertEqual(gate_error_manifest["candidate"]["status"], "rejected")
        self.assertIn("correctness gate error", gate_error_manifest["rejection_error"])

    def test_retry_in_flight_rejects_changed_candidate_artifact(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            checkpoint = root / "mutation-loop.json"
            candidate_dir = output / "op_a"
            candidate_dir.mkdir(parents=True)
            existing = candidate_dir / "existing.py"
            existing.write_text("def op_a(x):\n    return x\n", encoding="utf-8")

            def interrupted_generate(work, data, operator, parent, out, seed):
                (out / operator / "existing.py").write_text(
                    f"def {operator}(x):\n    return x + 2\n", encoding="utf-8"
                )
                raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                module.run_mutation_loop(
                    root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 1,
                    generate_fn=interrupted_generate, checkpoint_path=checkpoint,
                )
            generate = Mock()
            with self.assertRaisesRegex(ValueError, "ambiguous new artifacts"):
                module.run_mutation_loop(
                    root, datasets, output, "op_a", datasets / "op_a/op_a.py", 5, 1,
                    generate_fn=generate, checkpoint_path=checkpoint, resume=True,
                    retry_in_flight=True,
                )
            generate.assert_not_called()

    def test_checkpoint_lock_rejects_concurrent_writer(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            checkpoint = root / "mutation-loop.json"
            with module._checkpoint_writer_lock(checkpoint), self.assertRaisesRegex(
                RuntimeError, "already in use"
            ):
                module.run_mutation_loop(
                    root, datasets, root / "generated", "op_a",
                    datasets / "op_a/op_a.py", 5, 0, checkpoint_path=checkpoint,
                )

    def test_cli_rejects_invalid_resume_combinations_before_official_load(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.json"
            checkpoint.write_text("{}", encoding="utf-8")
            base = [
                "generate_official_candidates_batch.py",
                "--work-dir", str(root / "work"),
                "--datasets-dir", str(root / "datasets"),
                "--output-dir", str(root / "output"),
            ]
            cases = {
                "resume_without_loop": ["--resume", "--report", str(checkpoint)],
                "resume_without_report": [
                    "--mutation-loop", "--kernel", "op_a", "--resume"
                ],
                "retry_without_resume": [
                    "--mutation-loop", "--kernel", "op_a", "--retry-in-flight"
                ],
                "retry_without_call": [
                    "--mutation-loop", "--kernel", "op_a", "--resume",
                    "--retry-in-flight", "--report", str(checkpoint),
                ],
                "missing_checkpoint": [
                    "--mutation-loop", "--kernel", "op_a", "--resume",
                    "--max-new-calls", "1", "--report", str(root / "missing.json"),
                ],
                "fresh_existing_checkpoint": [
                    "--mutation-loop", "--kernel", "op_a", "--max-new-calls", "1",
                    "--report", str(checkpoint),
                ],
                "unpaired_budget": ["--remaining-token-budget", "1000"],
                "budget_without_loop": [
                    "--remaining-token-budget", "1000",
                    "--remaining-wall-seconds", "60",
                ],
                "loop_without_budget": [
                    "--mutation-loop", "--kernel", "op_a",
                ],
                "negative_calls": ["--max-new-calls", "-1"],
            }
            for label, extra in cases.items():
                with self.subTest(label=label), patch.object(
                    sys, "argv", base + extra
                ), patch.object(module, "_load_official_modules") as load, patch(
                    "sys.stderr", new=io.StringIO()
                ):
                    with self.assertRaises(SystemExit) as raised:
                        module.main()
                    self.assertEqual(raised.exception.code, 2)
                    load.assert_not_called()

    def test_cli_wires_explicit_retry_to_mutation_loop(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "op_a.py"
            parent.write_text("def op_a(x):\n    return x\n", encoding="utf-8")
            checkpoint = root / "checkpoint.json"
            checkpoint.write_text("{}", encoding="utf-8")
            argv = [
                "generate_official_candidates_batch.py",
                "--work-dir", str(root / "work"),
                "--datasets-dir", str(root / "datasets"),
                "--output-dir", str(root / "output"),
                "--mutation-loop", "--kernel", "op_a",
                "--resume", "--retry-in-flight",
                "--max-new-calls", "1", "--report", str(checkpoint),
                "--remaining-token-budget", "1000",
                "--remaining-wall-seconds", "60",
            ]
            report = {"last_run_status_counts": {}, "records": []}
            with patch.object(sys, "argv", argv), patch.object(
                module, "_load_official_modules", return_value=(None, None, object())
            ), patch.object(
                module, "_select_parent", return_value=(parent, "test-parent")
            ), patch.object(
                module, "_correctness_gate_for", return_value=None
            ), patch.object(
                module, "run_mutation_loop", return_value=report
            ) as run, patch("sys.stdout", new=io.StringIO()):
                result = module.main()

            self.assertEqual(result, 0)
            self.assertTrue(run.call_args.kwargs["resume"])
            self.assertTrue(run.call_args.kwargs["retry_in_flight"])
            budget = run.call_args.kwargs["budget_controller"]
            self.assertIsInstance(budget, BudgetController)
            self.assertEqual(budget.limits, BudgetLimits(1_000, 60))

    def test_no_correctness_gate_keeps_width_one_greedy_chain(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            calls = []

            def fake_generate(work, data, operator, parent, out, seed):
                ordinal = len(calls)
                calls.append(parent.name)
                candidate_id = f"child-{ordinal}"
                code = f"def {operator}(x):\n    return x + {ordinal + 2}\n"
                self._write_pass(out, operator, candidate_id, code)
                return {"candidate": {"id": candidate_id}}

            report = module.run_mutation_loop(
                root,
                datasets,
                output,
                "op_a",
                datasets / "op_a/op_a.py",
                random_seed=5,
                max_new_calls=2,
                generate_fn=fake_generate,
            )
            with self.assertRaisesRegex(ValueError, "requires a correctness gate"):
                module.run_mutation_loop(
                    root,
                    datasets,
                    output,
                    "op_a",
                    datasets / "op_a/op_a.py",
                    random_seed=5,
                    max_new_calls=0,
                    beam_width=2,
                )

        self.assertEqual(calls, ["op_a.py", "child-0.py"])
        self.assertEqual(
            [record["status"] for record in report["records"]],
            ["static_import_pass"] * 2,
        )
        self.assertTrue(
            report["final_admitted_parent_path"].endswith("generated/op_a/child-1.py")
        )

    def test_selective_gate_selection_is_bounded_and_fail_closed(self) -> None:
        module = load_batch_module()
        operator = "_selective_scan_update_kernel"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "generated"
            self._write_pass(output, operator, "candidate")
            manifest = output / operator / "candidate.manifest.json"
            completed = [
                SimpleNamespace(
                    returncode=0,
                    stdout=f"{module._SELECTIVE_CUDA_COMPLETION_MARKER} []",
                    stderr="",
                ),
                SimpleNamespace(returncode=1, stdout="", stderr="matrix failed"),
                SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="OK (skipped=1)",
                ),
            ]
            with patch.dict(
                os.environ, {"WLZ_TRITON_PYTHON": sys.executable}
            ), patch.object(module.subprocess, "run", side_effect=completed) as run:
                gate = module._correctness_gate_for(operator, root / "datasets")
                passed = gate(manifest)
                failed = gate(manifest)
                skipped = gate(manifest)

        self.assertTrue(passed["eligible_for_performance"])
        self.assertEqual(passed["status"], "passed")
        self.assertFalse(failed["eligible_for_performance"])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["blocking_reasons"], ["failed_case"])
        self.assertFalse(skipped["eligible_for_performance"])
        self.assertEqual(skipped["status"], "failed")
        self.assertFalse(skipped["results"][0]["matrix_completed"])
        self.assertEqual(run.call_count, 3)
        command = run.call_args_list[0].args[0]
        self.assertEqual(command[-2:], [module._SELECTIVE_CUDA_TEST, "-v"])
        self.assertEqual(run.call_args_list[0].kwargs["timeout"], 120)
        self.assertTrue(
            run.call_args_list[0].kwargs["env"]["WLZ_SELECTIVE_CANDIDATE"].endswith(
                "candidate.py"
            )
        )
        self.assertIsNone(module._correctness_gate_for("op_a", Path("datasets")))

    def test_quantize_gate_requires_completed_cuda_matrix(self) -> None:
        module = load_batch_module()
        operator = "_quantize_k_cache_fast_kernel"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "generated"
            self._write_pass(output, operator, "candidate")
            manifest = output / operator / "candidate.manifest.json"
            completed = [
                SimpleNamespace(
                    returncode=0,
                    stdout=f"{module._QUANTIZE_CUDA_COMPLETION_MARKER}\n",
                    stderr="",
                ),
                SimpleNamespace(returncode=1, stdout="", stderr="failed"),
                SimpleNamespace(returncode=0, stdout="", stderr="OK (skipped=1)"),
            ]
            with patch.dict(
                os.environ, {"WLZ_TRITON_PYTHON": sys.executable}
            ), patch.object(module.subprocess, "run", side_effect=completed) as run:
                gate = module._correctness_gate_for(operator, root / "datasets")
                passed = gate(manifest)
                failed = gate(manifest)
                skipped = gate(manifest)

            candidate_path = manifest.with_name("candidate.py")
            candidate_path.write_text("def changed():\n    pass\n", encoding="utf-8")
            invalid = gate(manifest)

        self.assertTrue(passed["eligible_for_performance"])
        self.assertEqual(passed["blocking_reasons"], [])
        self.assertFalse(failed["eligible_for_performance"])
        self.assertFalse(skipped["eligible_for_performance"])
        self.assertFalse(skipped["results"][0]["matrix_completed"])
        self.assertEqual(invalid["blocking_reasons"], ["invalid_candidate_manifest"])
        self.assertEqual(run.call_count, 3)
        self.assertEqual(
            run.call_args_list[0].kwargs["env"]["WLZ_QUANTIZE_K_CACHE_CANDIDATE"],
            str(candidate_path.resolve()),
        )
        self.assertEqual(
            run.call_args_list[0].kwargs["env"]["WLZ_CORRECTNESS_DATASETS_DIR"],
            str((root / "datasets").resolve()),
        )
        self.assertEqual(
            run.call_args_list[0].args[0][-2:], [module._QUANTIZE_CUDA_TEST, "-v"]
        )
        self.assertEqual(
            module._ADMISSION_POLICY_IDS[operator],
            "local-quantize-k-cache-public-cuda-v1",
        )
        self.assertEqual(
            passed["admission_policy_id"], module._ADMISSION_POLICY_IDS[operator]
        )

    def test_act_quant_gate_requires_completed_cuda_matrix(self) -> None:
        module = load_batch_module()
        operator = "_act_quant_kernel"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "generated"
            self._write_pass(output, operator, "candidate")
            manifest = output / operator / "candidate.manifest.json"
            completed = [
                SimpleNamespace(
                    returncode=0,
                    stdout=f"{module._ACT_QUANT_CUDA_COMPLETION_MARKER}\n",
                    stderr="",
                ),
                SimpleNamespace(returncode=0, stdout="", stderr="OK (skipped=1)"),
            ]
            with patch.dict(
                os.environ, {"WLZ_TRITON_PYTHON": sys.executable}
            ), patch.object(module.subprocess, "run", side_effect=completed) as run:
                gate = module._correctness_gate_for(operator, root / "datasets")
                passed = gate(manifest)
                skipped = gate(manifest)

        self.assertTrue(passed["eligible_for_performance"])
        self.assertFalse(skipped["eligible_for_performance"])
        self.assertFalse(skipped["results"][0]["matrix_completed"])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[0].kwargs["env"]["WLZ_ACT_QUANT_CANDIDATE"],
            str(manifest.with_name("candidate.py").resolve()),
        )
        self.assertEqual(
            run.call_args_list[0].kwargs["env"]["WLZ_CORRECTNESS_DATASETS_DIR"],
            str((root / "datasets").resolve()),
        )
        self.assertEqual(
            run.call_args_list[0].args[0][-2:], [module._ACT_QUANT_CUDA_TEST, "-v"]
        )
        self.assertEqual(
            module._ADMISSION_POLICY_IDS[operator],
            "local-act-quant-public-cuda-v1",
        )
        self.assertEqual(
            passed["admission_policy_id"], module._ADMISSION_POLICY_IDS[operator]
        )

    def test_set_k_and_s_gate_requires_completed_guarded_cuda_matrix(self) -> None:
        module = load_batch_module()
        operator = "_set_k_and_s_triton_kernel"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "generated"
            self._write_pass(output, operator, "candidate")
            manifest = output / operator / "candidate.manifest.json"
            completed = [
                SimpleNamespace(
                    returncode=0,
                    stdout=f"{module._SET_K_AND_S_CUDA_COMPLETION_MARKER}\n",
                    stderr="",
                ),
                SimpleNamespace(returncode=0, stdout="", stderr="OK (skipped=1)"),
            ]
            with patch.dict(
                os.environ, {"WLZ_TRITON_PYTHON": sys.executable}
            ), patch.object(module.subprocess, "run", side_effect=completed) as run:
                gate = module._correctness_gate_for(operator, root / "datasets")
                passed = gate(manifest)
                skipped = gate(manifest)

        self.assertTrue(passed["eligible_for_performance"])
        self.assertEqual(passed["evidence_scope"], "local_cuda_proxy_only_not_ascend_or_official")
        self.assertFalse(skipped["eligible_for_performance"])
        self.assertFalse(skipped["results"][0]["matrix_completed"])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[0].kwargs["env"]["WLZ_SET_K_AND_S_CANDIDATE"],
            str(manifest.with_name("candidate.py").resolve()),
        )
        self.assertEqual(
            run.call_args_list[0].kwargs["env"]["WLZ_CORRECTNESS_DATASETS_DIR"],
            str((root / "datasets").resolve()),
        )
        self.assertEqual(
            run.call_args_list[0].args[0][-2:], [module._SET_K_AND_S_CUDA_TEST, "-v"]
        )
        self.assertEqual(
            module._ADMISSION_POLICY_IDS[operator],
            "local-set-k-and-s-public-cuda-guard-v1",
        )
        self.assertEqual(
            passed["admission_policy_id"], module._ADMISSION_POLICY_IDS[operator]
        )

    def test_count_expert_gate_requires_completed_cuda_matrix(self) -> None:
        module = load_batch_module()
        operator = "_count_expert_num_tokens"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "generated"
            self._write_pass(output, operator, "candidate")
            manifest = output / operator / "candidate.manifest.json"
            completed = [
                SimpleNamespace(
                    returncode=0,
                    stdout=f"{module._COUNT_EXPERT_CUDA_COMPLETION_MARKER}\n",
                    stderr="",
                ),
                SimpleNamespace(returncode=0, stdout="", stderr="OK (skipped=1)"),
            ]
            with patch.dict(
                os.environ, {"WLZ_TRITON_PYTHON": sys.executable}
            ), patch.object(module.subprocess, "run", side_effect=completed) as run:
                gate = module._correctness_gate_for(operator, root / "datasets")
                passed = gate(manifest)
                skipped = gate(manifest)

        self.assertTrue(passed["eligible_for_performance"])
        self.assertEqual(passed["evidence_scope"], "local_cuda_proxy_only_not_ascend_or_official")
        self.assertFalse(skipped["eligible_for_performance"])
        self.assertFalse(skipped["results"][0]["matrix_completed"])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[0].kwargs["env"]["WLZ_COUNT_EXPERT_CANDIDATE"],
            str(manifest.with_name("candidate.py").resolve()),
        )
        self.assertEqual(
            run.call_args_list[0].kwargs["env"]["WLZ_CORRECTNESS_DATASETS_DIR"],
            str((root / "datasets").resolve()),
        )
        self.assertEqual(
            run.call_args_list[0].args[0][-2:], [module._COUNT_EXPERT_CUDA_TEST, "-v"]
        )
        self.assertEqual(
            module._ADMISSION_POLICY_IDS[operator],
            "local-count-expert-basic-no-map-cuda-v1",
        )
        self.assertEqual(
            passed["admission_policy_id"], module._ADMISSION_POLICY_IDS[operator]
        )

    def test_generated_parent_manifest_preserves_lineage(self) -> None:
        module = load_generator_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            parent_dir = output / "op_a"
            parent_dir.mkdir(parents=True)
            parent_code = "def op_a(x):\n    return x + 2\n"
            (parent_dir / "parent.py").write_text(parent_code, encoding="utf-8")
            (parent_dir / "parent.manifest.json").write_text(
                json.dumps(
                    {
                        "candidate": {
                            "id": "parent",
                            "op_name": "op_a",
                            "code_hash": sha256_text(parent_code),
                            "generation": 2,
                            "model_used": "parent-model",
                        }
                    }
                ),
                encoding="utf-8",
            )
            observed = {}

            def mutate(parent):
                observed.update(vars(parent))
                return SimpleNamespace(
                    code="def op_a(x):\n    return x + 3\n",
                    id="child",
                    generation=parent.generation + 1,
                    metadata={"mutation_type": "param_tuning"},
                    model_used="child-model",
                )

            fake_modules = (
                SimpleNamespace(
                    EAConfig=lambda: SimpleNamespace(
                        api_url=None, api_key=None, llm_models=["fake-model"]
                    )
                ),
                SimpleNamespace(
                    GeneticOperators=lambda *_: SimpleNamespace(mutate=mutate),
                    Individual=SimpleNamespace,
                ),
                SimpleNamespace(interface_contract_error=lambda *_: None),
            )
            fake_llm = SimpleNamespace(
                call_history=[{"prompt_sha256": "a" * 64}],
                get_stats=lambda: {"call_count": 1},
            )
            static_pass = SimpleNamespace(passed=True, to_dict=lambda: {"passed": True})
            imported = SimpleNamespace(
                status="imported",
                phase="module_import",
                error_type=None,
                error_message=None,
            )
            with patch.object(module, "_load_official_modules", return_value=fake_modules), patch.object(
                module, "StdlibOpenAIClient", return_value=fake_llm
            ), patch.object(module.LocalExecutor, "evaluate", return_value=static_pass), patch.object(
                module, "run_candidate", return_value=imported
            ):
                manifest = module.generate_candidate(
                    root,
                    datasets,
                    "op_a",
                    parent_dir / "parent.py",
                    output,
                    7,
                )

        self.assertEqual(observed["id"], "parent")
        self.assertEqual(observed["generation"], 2)
        self.assertEqual(observed["model_used"], "parent-model")
        self.assertEqual(manifest["candidate"]["parent_ids"], ["parent"])
        self.assertEqual(manifest["candidate"]["generation"], 3)

    def test_generator_attaches_safe_llm_stats_to_malformed_child(self) -> None:
        module = load_generator_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = self._dataset(root)
            output = root / "generated"
            stats = {
                "calls": [
                    {
                        "model": "deepseek-v4-pro",
                        "prompt_sha256": "e" * 64,
                        "status": "failed",
                    }
                ]
            }

            def malformed_child(parent):
                return SimpleNamespace(
                    code="def op_a(:\n    pass\n",
                    id="broken",
                    generation=1,
                    metadata={"mutation_type": "local_rewrite"},
                    model_used="deepseek-v4-pro",
                )

            fake_modules = (
                SimpleNamespace(
                    EAConfig=lambda: SimpleNamespace(
                        api_url=None, api_key=None, llm_models=["deepseek-v4-pro"]
                    )
                ),
                SimpleNamespace(
                    GeneticOperators=lambda *_: SimpleNamespace(mutate=malformed_child),
                    Individual=SimpleNamespace,
                ),
                SimpleNamespace(interface_contract_error=lambda *_: None),
            )
            fake_llm = SimpleNamespace(
                call_history=[{"prompt_sha256": "e" * 64}],
                get_stats=lambda: stats,
            )
            with patch.object(
                module, "_load_official_modules", return_value=fake_modules
            ), patch.object(module, "StdlibOpenAIClient", return_value=fake_llm):
                with self.assertRaises(SyntaxError) as raised:
                    module.generate_candidate(
                        root,
                        datasets,
                        "op_a",
                        datasets / "op_a/op_a.py",
                        output,
                        7,
                    )

        self.assertEqual(raised.exception._wlz_llm_stats, stats)

    def test_invalid_import_evidence_is_not_resumed(self) -> None:
        module = load_batch_module()
        for label, import_evaluation in (
            ("missing", None),
            ("null", None),
            ("failed", {"status": "import_error", "phase": "module_import"}),
            ("wrong_phase", {"status": "imported", "phase": "candidate_run"}),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                datasets = self._dataset(root)
                output = root / "generated"
                self._write_pass(output, "op_a", "old-a")
                manifest_path = output / "op_a/old-a.manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if label == "missing":
                    manifest.pop("import_evaluation")
                else:
                    manifest["import_evaluation"] = import_evaluation
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                report = module.run_batch(
                    root,
                    datasets,
                    output,
                    5,
                    0,
                    kernels=["op_a"],
                    dry_run=True,
                    contract_module=SimpleNamespace(interface_contract_error=lambda *_: None),
                )
            self.assertEqual(report["status_counts"], {"planned": 1})

    def test_parent_preflight_falls_back_from_cuda_variant(self) -> None:
        module = load_batch_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator_dir = root / "datasets" / "state_op"
            operator_dir.mkdir(parents=True)
            baseline = "def state_op(x):\n    with torch.npu.device(x.device.index):\n        return x\n"
            variant = baseline.replace("torch.npu.device", "torch.cuda.device")
            (operator_dir / "state_op.py").write_text(baseline, encoding="utf-8")
            (operator_dir / "state_op_1.py").write_text(variant, encoding="utf-8")

            parent, policy = module._select_parent(
                root / "datasets",
                "state_op",
                SimpleNamespace(interface_contract_error=lambda *_: None),
            )

        self.assertEqual(parent.name, "state_op.py")
        self.assertEqual(policy, "baseline_after_target_device_preflight")


if __name__ == "__main__":
    unittest.main()

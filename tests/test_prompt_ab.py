import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_prompt_ab.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prompt_ab_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PromptAbTests(unittest.TestCase):
    def test_schedule_is_paired_balanced_and_deterministic(self) -> None:
        module = load_module()
        schedule = module.build_schedule(2)
        self.assertEqual(schedule, module.build_schedule(2))
        self.assertEqual(len(schedule), 12)
        self.assertEqual(sum(item[3] == "A" for item in schedule), 6)
        self.assertEqual(sum(item[3] == "B" for item in schedule), 6)
        for repeat in range(2):
            for family, operator in module.CASES:
                variants = [
                    item[3] for item in schedule
                    if item[:3] == (repeat, family, operator)
                ]
                self.assertEqual(set(variants), {"A", "B"})

    def test_assessment_uses_production_style_cleaning_before_contract(self) -> None:
        module = load_module()
        parent = "def kernel(x):\n    return x\n"
        raw = "```python\ndef kernel(x):\n    return x + 1\n```"

        def cleaner(code):
            return code.replace("```python\n", "").replace("```", "")

        result = module.assess_candidate(raw, parent, cleaner, lambda _a, _b: None)
        self.assertFalse(result["raw_protocol_ok"])
        self.assertTrue(result["cleaned_syntax_ok"])
        self.assertTrue(result["interface_contract_ok"])
        self.assertTrue(result["non_original_structure"])

    def test_invalid_repeat_count_fails_closed(self) -> None:
        module = load_module()
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.build_schedule(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.build_repair_schedule(value)

    def test_repair_schedule_is_paired_and_alternating(self) -> None:
        module = load_module()
        self.assertEqual(
            module.build_repair_schedule(3),
            (
                (0, "A"),
                (0, "B"),
                (1, "B"),
                (1, "A"),
                (2, "A"),
                (2, "B"),
            ),
        )

    def test_variant_failure_stops_only_that_variant(self) -> None:
        module = load_module()

        class FakeClient:
            def __init__(self, fail: bool) -> None:
                self.fail = fail
                self.calls = 0
                self.call_history = []

            def generate(self, prompt, **kwargs):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("LLM HTTP 429: rate limited " + "x" * 300)
                self.call_history.append({
                    "prompt_sha256": "p", "request_fingerprint": "f",
                    "response_model": "m", "usage": {"total_tokens": 1},
                    "latency_seconds": 0.0,
                })
                return "def kernel(x):\n    return x + 1\n"

        limits = module.BudgetLimits(1000, 10)
        controllers = {name: module.BudgetController(limits) for name in ("A", "B")}
        clients = {"A": FakeClient(fail=True), "B": FakeClient(fail=False)}
        args = SimpleNamespace(
            work_dir=ROOT / "work" / "official_triton_agent",
            repeats=2, model="test-model", temperature=0.7,
        )
        assessment = {
            "raw_protocol_ok": True, "cleaned_syntax_ok": True,
            "interface_contract_ok": True, "non_original_structure": True,
            "candidate_sha256": "c",
        }
        with patch.object(module, "_clients", return_value=(controllers, clients)), \
                patch.object(module, "assess_candidate", return_value=assessment):
            result = module.run(args)

        stopped = [r for r in result["records"] if r.get("status") == "stopped"]
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0]["variant"], "A")
        self.assertEqual(len(stopped[0]["error"]), 200)
        self.assertTrue(stopped[0]["error"].startswith("LLM HTTP 429: rate limited"))
        self.assertEqual(clients["A"].calls, 1)
        self.assertEqual(clients["B"].calls, 6)
        b_records = [
            r for r in result["records"]
            if r["variant"] == "B" and r.get("status") != "stopped"
        ]
        self.assertEqual(len(b_records), 6)
        self.assertEqual(result["summary"]["A"]["calls"], 0)
        self.assertEqual(result["summary"]["B"]["calls"], 6)

    def test_p1_variant_removes_only_cross_layer_repetition(self) -> None:
        module = load_module()
        _, operators, _ = module._load_official_modules(
            (ROOT / "work" / "official_triton_agent").resolve()
        )
        parent = "def kernel(x):\n    return x\n"
        legacy_system, legacy_user, legacy_metadata = module._legacy_p0_prompt(
            parent, operators
        )
        skill = operators.MUTATION_SKILLS["param_tuning"]
        plan = operators.create_mutation_plan("seed-parent", parent, skill)
        current = operators.render_mutation_prompt(
            parent,
            parent_fitness=1.0,
            parent_model="provided-seed",
            skill=skill,
            mutation_plan=plan,
        )

        self.assertEqual(legacy_system, current.system_message)
        self.assertLess(len(current.user_message), len(legacy_user))
        self.assertNotIn("You are an expert in Triton", current.user_message)
        self.assertNotIn("Task: Apply the mutation", current.user_message)
        self.assertNotIn("Generate the mutated kernel:", current.user_message)
        for required in (
            *operators.INTERFACE_CONTRACT_RULES,
            *skill.instructions,
            "BEGIN PARENT SOURCE",
            "Mutation Plan Version",
            "make at least one executable change",
            "Output ONLY the code",
        ):
            self.assertIn(required, current.user_message)
        self.assertEqual(
            legacy_metadata["mutation_prompt_version"],
            module.LEGACY_MUTATION_PROMPT_VERSION,
        )
        self.assertNotEqual(
            legacy_metadata["mutation_prompt_version"],
            operators.MUTATION_PROMPT_VERSION,
        )


if __name__ == "__main__":
    unittest.main()

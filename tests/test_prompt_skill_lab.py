import hashlib
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_FILE = ROOT / "work" / "prompt_skill_lab.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prompt_skill_lab_test", MODULE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {MODULE_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PromptSkillLabTests(unittest.TestCase):
    def test_system_prompt_priority_matches_doge_contract(self) -> None:
        module = load_module()
        defaults = ("base-a", "base-b")
        self.assertEqual(
            module.build_effective_system_prompt(defaults, append_section="tail"),
            ("base-a", "base-b", "tail"),
        )
        self.assertEqual(
            module.build_effective_system_prompt(
                defaults, custom_section="custom", append_section="tail"
            ),
            ("custom", "tail"),
        )
        self.assertEqual(
            module.build_effective_system_prompt(
                defaults,
                custom_section="custom",
                append_section="tail",
                override_section="override",
            ),
            ("override",),
        )

    def test_discovery_is_bounded_and_does_not_include_skill_content(self) -> None:
        module = load_module()
        skills = (
            module.PromptSkill(
                "param-tuning", "v1", "Tune general launch parameters", "For tile tuning", "FULL-PARAM-CONTENT"
            ),
            module.PromptSkill(
                "local-rewrite", "v2", "Rewrite one bounded fragment", "For local changes", "FULL-LOCAL-CONTENT"
            ),
        )
        catalog = module.SkillCatalog(skills)
        full_listing = catalog.discovery_listing()
        self.assertIn("param-tuning", full_listing)
        self.assertIn("For tile tuning", full_listing)
        self.assertNotIn("FULL-PARAM-CONTENT", full_listing)
        self.assertNotIn("FULL-LOCAL-CONTENT", full_listing)

        bounded = catalog.discovery_listing(char_budget=90)
        self.assertLessEqual(len(bounded), 90)
        self.assertIn("param-tuning", bounded)
        self.assertIn("local-rewrite", bounded)

    def test_activation_returns_only_the_selected_versioned_content(self) -> None:
        module = load_module()
        catalog = module.SkillCatalog((
            module.PromptSkill("first", "first-v1", "First skill", "", "FIRST-FULL"),
            module.PromptSkill("second", "second-v3", "Second skill", "", "SECOND-FULL"),
            module.PromptSkill("hidden", "hidden-v1", "Hidden skill", "", "HIDDEN", False),
        ))

        activated = catalog.activate("second")
        self.assertEqual(
            (activated.name, activated.version, activated.content),
            ("second", "second-v3", "SECOND-FULL"),
        )
        self.assertNotIn("HIDDEN", catalog.discovery_listing())
        with self.assertRaises(KeyError):
            catalog.activate("hidden")
        with self.assertRaises(ValueError):
            module.SkillCatalog((
                module.PromptSkill("same", "v1", "One", "", "ONE"),
                module.PromptSkill("same", "v2", "Two", "", "TWO"),
            ))

    def test_generic_catalog_is_deterministic_and_has_no_competition_registry(self) -> None:
        module = load_module()
        skill = module.PromptSkill(
            "example", "example-v1", "Example skill", "For testing", "EXAMPLE"
        )
        catalog = module.SkillCatalog((skill,))

        self.assertEqual(catalog.activate("example"), catalog.activate("example"))
        self.assertEqual(
            hashlib.sha256(catalog.activate("example").content.encode("utf-8")).hexdigest(),
            hashlib.sha256(b"EXAMPLE").hexdigest(),
        )
        with self.assertRaises(KeyError):
            catalog.activate("unknown")
        self.assertFalse(hasattr(module, "COMPETITION_SKILL_CATALOG"))
        self.assertFalse(hasattr(module, "render_competition_mutation_prompt"))


if __name__ == "__main__":
    unittest.main()

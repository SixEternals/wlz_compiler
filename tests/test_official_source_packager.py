import importlib.util
import ast
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_official_agent_source_smoke.py"


def load_packager():
    spec = importlib.util.spec_from_file_location("official_source_packager_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialSourcePackagerTests(unittest.TestCase):
    def test_builds_deterministic_audited_agent_root(self):
        module = load_packager()
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.zip"
            second = Path(tmp) / "second.zip"
            first_manifest = module.build_source_smoke(first)
            second_manifest = module.build_source_smoke(second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                first_manifest["artifact_sha256"], second_manifest["artifact_sha256"]
            )
            self.assertFalse(first_manifest["official_scoring_ready"])
            self.assertEqual(first_manifest["effective_config"], {
                "population_size": 2,
                "max_generations": 0,
                "max_total_tokens": 8192,
            })
            self.assertEqual(first_manifest["expected_entrypoint"], "Agent/main.py")
            with zipfile.ZipFile(first) as archive:
                names = archive.namelist()
                self.assertIsNone(archive.testzip())
                self.assertIn("Agent/main.py", names)
                self.assertIn("Agent/baseline/baseline.json", names)
                self.assertIn("Agent/wlz_optimizer/budget.py", names)
                config = ast.parse(archive.read("Agent/config.py"))
                defaults = {
                    statement.target.id: statement.value.value
                    for node in config.body
                    if isinstance(node, ast.ClassDef) and node.name == "EAConfig"
                    for statement in node.body
                    if (
                        isinstance(statement, ast.AnnAssign)
                        and isinstance(statement.target, ast.Name)
                        and statement.target.id in module.SMOKE_OVERRIDES
                    )
                }
                self.assertEqual(defaults, module.SMOKE_OVERRIDES)
                self.assertFalse(any("datasets" in Path(name).parts for name in names))
                self.assertFalse(any("__pycache__" in name for name in names))
            sidecar = json.loads(
                first.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar, first_manifest)

    def test_rejects_credentials_unsafe_paths_and_overwrite(self):
        module = load_packager()
        with self.assertRaisesRegex(ValueError, "API key"):
            module._validate_source("Agent/config.py", b"key='sk-" + b"a" * 24 + b"'")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            module._validate_source("../config.py", b"pass\n")
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "source.zip"
            module.build_source_smoke(artifact)
            with self.assertRaises(FileExistsError):
                module.build_source_smoke(artifact)


if __name__ == "__main__":
    unittest.main()

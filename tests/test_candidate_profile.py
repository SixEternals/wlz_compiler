"""Focused tests for Candidate tile/block/launch metadata."""

import unittest

from wlz_optimizer.schemas import Candidate, LaunchProfile, TileDimBinding


HASH_A = "a" * 64
HASH_B = "b" * 64


def binding(**overrides) -> TileDimBinding:
    values = {
        "parameter_name": "BLOCK_SIZE_M",
        "symbolic_dimension": "M",
        "value": 128,
        "source_kind": "static_ast",
        "source_ref": "kernel.py:12",
    }
    values.update(overrides)
    return TileDimBinding(**values)


def profile(**overrides) -> LaunchProfile:
    values = {
        "candidate_code_hash": HASH_A,
        "source_kind": "static_ast",
        "source_ref": "kernel.py:10-20",
        "tiles": [binding()],
        "num_warps": 4,
        "num_stages": 2,
        "grid_rank": 2,
    }
    values.update(overrides)
    return LaunchProfile(**values)


def candidate(**overrides) -> Candidate:
    values = {
        "id": "candidate-1",
        "op_name": "matmul",
        "code": "def kernel(): pass\n",
        "code_hash": HASH_A,
        "parent_ids": [],
        "generation": 0,
        "mutation_kind": "baseline",
        "model_used": None,
        "prompt_id": None,
        "status": "created",
        "score": None,
        "metadata": {},
    }
    values.update(overrides)
    return Candidate(**values)


class TileDimBindingTests(unittest.TestCase):
    def test_records_traceable_tile_source(self) -> None:
        item = binding()
        self.assertEqual(item.parameter_name, "BLOCK_SIZE_M")
        self.assertEqual(item.symbolic_dimension, "M")
        self.assertEqual(item.value, 128)
        self.assertTrue(item.trusted_for_boundary_proposals)
        self.assertEqual(TileDimBinding.from_dict(item.to_dict()), item)

    def test_model_declared_value_is_explicitly_untrusted(self) -> None:
        item = binding(source_kind="model_declared", source_ref="prompt:mutation-7")
        self.assertFalse(item.trusted_for_boundary_proposals)

    def test_rejects_invalid_binding_fields(self) -> None:
        invalid = (
            {"parameter_name": ""},
            {"symbolic_dimension": ""},
            {"value": 0},
            {"value": True},
            {"source_kind": "unknown"},
            {"source_ref": ""},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                binding(**values)


class LaunchProfileTests(unittest.TestCase):
    def test_roundtrip_preserves_tile_and_launch_metadata(self) -> None:
        original = profile()
        restored = LaunchProfile.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.schema_version, 1)
        self.assertTrue(restored.trusted_for_boundary_proposals)

    def test_model_declared_profile_is_explicitly_untrusted(self) -> None:
        item = profile(source_kind="model_declared", source_ref="prompt:mutation-7")
        self.assertFalse(item.trusted_for_boundary_proposals)

    def test_rejects_duplicate_tile_parameter_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            profile(
                tiles=[
                    binding(),
                    binding(symbolic_dimension="N", value=64, source_ref="kernel.py:13"),
                ]
            )

    def test_rejects_invalid_hash_and_schema(self) -> None:
        with self.assertRaises(ValueError):
            profile(candidate_code_hash="bad")
        with self.assertRaises(ValueError):
            profile(schema_version=2)

    def test_rejects_invalid_launch_values(self) -> None:
        invalid = (
            {"num_warps": 0},
            {"num_stages": 0},
            {"grid_rank": 0},
            {"grid_rank": 4},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                profile(**values)


class CandidateLaunchProfileTests(unittest.TestCase):
    def test_existing_candidate_without_profile_remains_valid(self) -> None:
        item = candidate()
        self.assertIsNone(item.launch_profile)
        self.assertIsNone(item.to_dict()["launch_profile"])

    def test_matching_profile_is_serialized_independently(self) -> None:
        item = candidate(launch_profile=profile())
        serialized = item.to_dict(include_code=False)
        self.assertNotIn("code", serialized)
        self.assertEqual(
            serialized["launch_profile"]["candidate_code_hash"],
            item.code_hash,
        )
        self.assertEqual(serialized["launch_profile"]["tiles"][0]["value"], 128)

    def test_stale_profile_is_rejected_by_candidate_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            candidate(launch_profile=profile(candidate_code_hash=HASH_B))


if __name__ == "__main__":
    unittest.main()

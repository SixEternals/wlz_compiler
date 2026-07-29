#!/usr/bin/env python3
"""Generate one real organizer-style mutation and stop unless static gates pass."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import random
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.candidate_runner import CandidateRunRequest, run_candidate
from wlz_optimizer.budget import BudgetController, BudgetLimits
from wlz_optimizer.cache import OfficialFailureHistory
from wlz_optimizer.executors import LocalExecutor
from wlz_optimizer.hash_utils import sha256_text
from wlz_optimizer.io_utils import load_operator_input
from wlz_optimizer.repair_guidance import (
    REPAIR_POLICY_VERSION,
    decide_official_repair,
)
from wlz_optimizer.schemas import Candidate, EvalContext
from wlz_optimizer.stdlib_llm import StdlibOpenAIClient


MUTATION_TYPES = ("param_tuning", "strategy_change", "local_rewrite")
_NESTED_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _direct_scope_nodes(root: ast.AST):
    for child in ast.iter_child_nodes(root):
        if isinstance(child, _NESTED_SCOPES):
            continue
        yield child
        yield from _direct_scope_nodes(child)


def _normalize_direct_local_names(tree: ast.AST) -> None:
    for function in [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]:
        nodes = list(_direct_scope_nodes(function))
        parameters = {
            arg.arg
            for arg in (
                list(function.args.posonlyargs)
                + list(function.args.args)
                + list(function.args.kwonlyargs)
                + ([function.args.vararg] if function.args.vararg else [])
                + ([function.args.kwarg] if function.args.kwarg else [])
            )
        }
        excluded = {
            name
            for node in nodes
            if isinstance(node, (ast.Global, ast.Nonlocal))
            for name in node.names
        }
        local_names: list[str] = []
        for node in nodes:
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and node.id not in parameters
                and node.id not in excluded
                and node.id not in local_names
            ):
                local_names.append(node.id)
        replacements = {
            name: f"<local:{index}>" for index, name in enumerate(local_names)
        }
        for node in nodes:
            if isinstance(node, ast.Name) and node.id in replacements:
                node.id = replacements[node.id]


def _load_official_modules(work_dir: Path):
    sys.path.insert(0, str(work_dir))
    shim = types.ModuleType("llm_interface")
    shim.LLMInterface = StdlibOpenAIClient
    sys.modules["llm_interface"] = shim
    config_module = importlib.import_module("config")
    operators_module = importlib.import_module("genetic_operators")
    contract_module = importlib.import_module("contract_executor")
    return config_module, operators_module, contract_module


def _structure_hash(code: str) -> str:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            node.msg = None
        elif (
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id in {"ValueError", "TypeError", "RuntimeError"}
            and not node.exc.keywords
            and all(isinstance(arg, (ast.Constant, ast.JoinedStr)) for arg in node.exc.args)
        ):
            node.exc.args = [ast.Constant(value="<diagnostic>")]
        elif isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node, clean=False) is not None:
                node.body = node.body[1:]
    _normalize_direct_local_names(tree)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _manifest_path(path: Path) -> str:
    """Prefer repository-relative provenance paths without rejecting external outputs."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_repair_guidance(
    manual_guidance: str | None,
    history_path: Path | None,
    observation_id: str | None,
    kernel: str,
    parent_path: Path,
    *,
    prior_repair_attempts: int = 0,
    budget_controller: BudgetController | None = None,
    estimated_total_tokens: int = 16_384,
    expected_seconds: float = 120.0,
) -> str | None:
    if (
        isinstance(prior_repair_attempts, bool)
        or not isinstance(prior_repair_attempts, int)
        or prior_repair_attempts < 0
    ):
        raise ValueError("prior_repair_attempts must be a non-negative integer")
    if (history_path is None) != (observation_id is None):
        raise ValueError(
            "--failure-history and --failure-observation-id must be supplied together"
        )
    if history_path is None:
        if manual_guidance is not None and prior_repair_attempts >= 1:
            raise ValueError("Repair denied: repair_attempt_limit")
        return manual_guidance
    if manual_guidance is not None:
        raise ValueError("Official failure history cannot be combined with --repair-guidance")
    if budget_controller is None:
        raise ValueError("Official failure repair requires an explicit remaining budget")
    decision = decide_official_repair(
        OfficialFailureHistory(history_path),
        operator=kernel,
        candidate_code_hash=sha256_text(parent_path.read_text(encoding="utf-8")),
        observation_id=observation_id,
        prior_repair_attempts=prior_repair_attempts,
        budget=budget_controller,
        estimated_total_tokens=estimated_total_tokens,
        expected_seconds=expected_seconds,
    )
    if not decision.allowed:
        raise ValueError(f"Repair denied: {decision.reason}")
    return decision.guidance


def _parent_repair_attempts(parent_path: Path) -> int:
    manifest_path = parent_path.with_name(f"{parent_path.stem}.manifest.json")
    if not manifest_path.is_file():
        return 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest["candidate"].get("metadata", {})
        official = metadata.get("official_operator_metadata", {})
        attempts = official.get("repair_attempt_count", 0)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Parent manifest has invalid repair provenance") from exc
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
        raise ValueError("Parent manifest has invalid repair_attempt_count")
    return attempts


def generate_candidate(
    work_dir: Path,
    datasets_dir: Path,
    kernel: str,
    parent_path: Path,
    output_dir: Path,
    random_seed: int,
    repair_guidance: str | None = None,
    mutation_type: str | None = None,
    budget_controller: BudgetController | None = None,
    repair_policy_version: str | None = None,
) -> dict:
    config_module, operators_module, contract_module = _load_official_modules(work_dir.resolve())
    config = config_module.EAConfig()
    config.api_url = os.environ.get("API_URL")
    config.api_key = os.environ.get("API_KEY")
    config.llm_models = [os.environ.get("ENGINE", config.llm_models[0])]
    if budget_controller is not None:
        config.budget_controller = budget_controller

    operator_input = load_operator_input(datasets_dir, kernel)
    parent_path = parent_path.resolve()
    allowed_parent_dirs = {
        operator_input.op_dir.resolve(),
        (output_dir / kernel).resolve(),
    }
    if parent_path.parent not in allowed_parent_dirs:
        raise ValueError(
            "Parent must be inside the selected operator or generated candidate directory"
        )
    parent_code = parent_path.read_text(encoding="utf-8")
    parent_hash = sha256_text(parent_code)
    parent_id = f"seed-{parent_hash[:12]}"
    parent_generation = 0
    parent_model = "provided-seed"
    prior_repair_attempts = _parent_repair_attempts(parent_path)
    if parent_path.parent == (output_dir / kernel).resolve():
        parent_manifest_path = parent_path.with_name(
            f"{parent_path.stem}.manifest.json"
        )
        try:
            parent_record = json.loads(parent_manifest_path.read_text(encoding="utf-8"))[
                "candidate"
            ]
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Generated parent requires a valid adjacent manifest") from exc
        if (
            parent_record.get("id") != parent_path.stem
            or parent_record.get("op_name") != kernel
            or parent_record.get("code_hash") != parent_hash
            or not isinstance(parent_record.get("generation"), int)
        ):
            raise ValueError("Generated parent manifest does not match its candidate")
        parent_id = parent_record["id"]
        parent_generation = parent_record["generation"]
        parent_model = parent_record.get("model_used") or "provided-parent"
    if repair_guidance and prior_repair_attempts >= 1:
        raise ValueError("Repair denied: repair_attempt_limit")

    random.seed(random_seed)
    llm = StdlibOpenAIClient(config)
    operators = operators_module.GeneticOperators(llm, config)
    parent_metadata = {}
    if isinstance(repair_guidance, str) and repair_guidance.strip():
        parent_metadata["repair_guidance"] = repair_guidance.strip()
    if mutation_type is not None:
        parent_metadata["mutation_type_override"] = mutation_type
    parent = operators_module.Individual(
        code=parent_code,
        generation=parent_generation,
        id=parent_id,
        model_used=parent_model,
        metadata=parent_metadata,
    )
    try:
        child = operators.mutate(parent)
        child.metadata["repair_attempt_count"] = prior_repair_attempts + int(
            bool(repair_guidance)
        )
        if repair_guidance and repair_policy_version is not None:
            child.metadata["repair_policy_version"] = repair_policy_version
        candidate_code = child.code.rstrip() + "\n"
        candidate_hash = sha256_text(candidate_code)
        seed_hashes = {sha256_text(seed["code"]) for seed in operator_input.seeds}
        seed_structures = {
            _structure_hash(seed["code"]) for seed in operator_input.seeds
        }
        candidate_structure = _structure_hash(candidate_code)
        candidate = Candidate(
            id=child.id,
            op_name=kernel,
            code=candidate_code,
            code_hash=candidate_hash,
            parent_ids=[parent_id],
            generation=child.generation,
            mutation_kind=child.metadata.get("mutation_type", "mutation"),
            model_used=child.model_used,
            prompt_id=llm.call_history[-1]["prompt_sha256"],
            status="generated",
            score=None,
            metadata={"official_operator_metadata": child.metadata},
        )
    except Exception as exc:
        # Preserve safe request provenance when mutation fails before a manifest exists.
        exc._wlz_llm_stats = llm.get_stats()
        raise
    rejection_error = None
    if candidate_hash in seed_hashes:
        rejection_error = "Generated candidate is byte-identical to an input seed"
    elif candidate_structure in seed_structures:
        rejection_error = "Generated candidate has no AST-level change from input seeds"
    elif candidate_hash == parent_hash:
        rejection_error = "Generated candidate is byte-identical to its parent"
    else:
        contract_error = contract_module.interface_contract_error(
            operator_input.baseline_file.read_text(encoding="utf-8"), candidate_code
        )
        if contract_error is not None:
            rejection_error = (
                f"Generated candidate failed full interface contract: {contract_error}"
            )

    context = EvalContext(
        op_name=kernel,
        input_dir=datasets_dir,
        output_dir=output_dir,
        required_functions=operator_input.required_functions,
        test_file=operator_input.test_file,
        baseline_file=operator_input.baseline_file,
    )
    static_result = None
    import_result = None
    if rejection_error is None:
        static_result = LocalExecutor().evaluate(candidate, context)
        if not static_result.passed:
            rejection_error = (
                f"Generated candidate failed static gate: {static_result.error_type}: "
                f"{static_result.error_message}"
            )
    if rejection_error is None:
        import_result = run_candidate(
            CandidateRunRequest(
                candidate_id=candidate.id,
                code=candidate_code,
                code_hash=candidate_hash,
                entrypoint=(operator_input.required_functions or [kernel])[0],
                payload={},
                timeout_seconds=20.0,
                stop_after_import=True,
            )
        )
        if (import_result.status, import_result.phase) != ("imported", "module_import"):
            rejection_error = (
                "Generated candidate failed import gate: "
                f"{import_result.status}/{import_result.phase}: "
                f"{import_result.error_type}: {import_result.error_message}"
            )[:600]

    candidate_dir = output_dir / kernel
    candidate_path = candidate_dir / f"{candidate.id}.py"
    manifest_path = candidate_dir / f"{candidate.id}.manifest.json"
    if candidate_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing generated candidate")
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate.status = "rejected" if rejection_error else "static_pass"
    candidate_path.write_text(candidate_code, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_kind": "real-agent-candidate",
        "candidate": candidate.to_dict(include_code=False),
        "candidate_path": _manifest_path(candidate_path),
        "parent_path": _manifest_path(parent_path),
        "parent_sha256": parent_hash,
        "seed_sha256": sorted(seed_hashes),
        "static_evaluation": static_result.to_dict() if static_result else None,
        "import_evaluation": (
            {
                "status": import_result.status,
                "phase": import_result.phase,
                "error_type": import_result.error_type,
                "error_message": import_result.error_message,
            }
            if import_result
            else None
        ),
        "rejection_error": rejection_error,
        "llm_stats": llm.get_stats(),
        "random_seed": random_seed,
        "repair_guidance_sha256": child.metadata.get("repair_guidance_sha256"),
        "repair_attempt_count": child.metadata["repair_attempt_count"],
        "repair_policy_version": child.metadata.get("repair_policy_version"),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if rejection_error is not None:
        raise ValueError(f"{rejection_error}; rejected candidate saved at {candidate_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-seed", type=int, default=20260712)
    parser.add_argument(
        "--repair-guidance",
        default=None,
        help="Optional explicit repair constraint appended to the mutation prompt",
    )
    parser.add_argument(
        "--mutation-type",
        choices=MUTATION_TYPES,
        help="Use one explicit mutation strategy instead of random selection",
    )
    parser.add_argument("--failure-history", type=Path)
    parser.add_argument("--failure-observation-id")
    parser.add_argument("--remaining-token-budget", type=int)
    parser.add_argument("--remaining-wall-seconds", type=float)
    parser.add_argument("--repair-request-token-upper-bound", type=int, default=16_384)
    parser.add_argument("--repair-expected-seconds", type=float, default=120.0)
    args = parser.parse_args()
    budget_controller = None
    if args.failure_history is not None:
        if args.remaining_token_budget is None or args.remaining_wall_seconds is None:
            parser.error(
                "--failure-history requires --remaining-token-budget and "
                "--remaining-wall-seconds"
            )
        try:
            budget_controller = BudgetController(BudgetLimits(
                args.remaining_token_budget,
                args.remaining_wall_seconds,
            ))
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
    prior_repair_attempts = _parent_repair_attempts(Path(args.parent))
    try:
        repair_guidance = _resolve_repair_guidance(
            args.repair_guidance,
            args.failure_history,
            args.failure_observation_id,
            args.kernel,
            Path(args.parent),
            prior_repair_attempts=prior_repair_attempts,
            budget_controller=budget_controller,
            estimated_total_tokens=args.repair_request_token_upper_bound,
            expected_seconds=args.repair_expected_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    manifest = generate_candidate(
        Path(args.work_dir),
        Path(args.datasets_dir),
        args.kernel,
        Path(args.parent),
        Path(args.output_dir),
        args.random_seed,
        repair_guidance,
        args.mutation_type,
        budget_controller=budget_controller,
        repair_policy_version=(
            REPAIR_POLICY_VERSION if args.failure_history is not None else None
        ),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

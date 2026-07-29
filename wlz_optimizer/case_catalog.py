"""Evidence-only catalog of currently visible public test scripts."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .schemas import (
    ArgumentBinding,
    EvaluationCase,
    ExecutionBinding,
    InputContract,
    OraclePolicy,
    OracleTarget,
    TensorInitializer,
    TensorInputContract,
    ValueSelector,
)


IDENTITY_SIGNATURE_KIND = "public-test-identity-v1"
ACT_QUANT_CASE_PATH = "_act_quant_kernel/test__act_quant_kernel_1.py"
ACT_QUANT_CASE_SOURCE_SHA256 = (
    "2ab41625c1f7b3d9e477569cfe536357f1af3808b00ec9ae9a3ef4c978e59339"
)
ACT_QUANT_DEFAULT_REFERENCE_ID = f"{ACT_QUANT_CASE_PATH}::ref_act_quant"
ACT_QUANT_ROUND_REFERENCE_ID = f"{ACT_QUANT_CASE_PATH}::round_expected_shapes"
CHUNK_CUMSUM_CASE_PATH = (
    "_chunk_cumsum_fwd_kernel/test__chunk_cumsum_fwd_kernel_1.py"
)
CHUNK_CUMSUM_CASE_SOURCE_SHA256 = (
    "e7b0fa32682e78fefbac446b3d3bcf1ef95c826c9032dceb8c45fd5946251dd8"
)
CHUNK_CUMSUM_REFERENCE_ID = f"{CHUNK_CUMSUM_CASE_PATH}::chunk_cumsum_ref"
COUNT_EXPERT_CASE_PATH = (
    "_count_expert_num_tokens/test__count_expert_num_tokens_1.py"
)
COUNT_EXPERT_CASE_SOURCE_SHA256 = (
    "a1617793fd39900448678ac1a8da7b93926f67f4ad83edfa5f2547ae3521e36d"
)
COUNT_EXPERT_REFERENCE_ID = f"{COUNT_EXPERT_CASE_PATH}::basic_no_map"
QUANTIZE_K_CACHE_CASE_PATH = (
    "_quantize_k_cache_fast_kernel/test__quantize_k_cache_fast_kernel_1.py"
)
QUANTIZE_K_CACHE_CASE_SOURCE_SHA256 = (
    "c07e3c50d0099a4725d12699f45c56c6105afae7ca08a1ccc230d7c2b4fbbec2"
)
QUANTIZE_K_CACHE_ROPE_REFERENCE_ID = f"{QUANTIZE_K_CACHE_CASE_PATH}::k_rope_input"
QUANTIZE_K_CACHE_METADATA_REFERENCE_ID = (
    f"{QUANTIZE_K_CACHE_CASE_PATH}::output_shape_4x592_bfloat16_npu"
)
SET_K_AND_S_CASE_PATH = (
    "_set_k_and_s_triton_kernel/test__set_k_and_s_triton_kernel_1.py"
)
SET_K_AND_S_CASE_SOURCE_SHA256 = (
    "5bebf3cc79f77add34f95e6065131a990fde9a389517372ddb6722b2341df99b"
)
SET_K_AND_S_REFERENCE_ID = f"{SET_K_AND_S_CASE_PATH}::first_token_raw_bytes"
PER_GROUP_TRANSPOSE_CASE_PATH = (
    "_per_group_transpose/test__per_group_transpose_1.py"
)
PER_GROUP_TRANSPOSE_CASE_SOURCE_SHA256 = (
    "1a905ceb470c8584771ec6558222b5852f79d94103c17121b37864f7ee0b6ce1"
)
PER_GROUP_TRANSPOSE_REFERENCE_ID = (
    f"{PER_GROUP_TRANSPOSE_CASE_PATH}::blockwise_transpose"
)
PACK_SEQ_CASE_PATH = "_pack_seq_kernel/test__pack_seq_kernel_1.py"
PACK_SEQ_CASE_SOURCE_SHA256 = (
    "0fcace02991de80bbe981ba73d83c215070d8924f984ed02da0401edc7e2fca0"
)
PACK_SEQ_REFERENCE_ID = f"{PACK_SEQ_CASE_PATH}::pack_seq_reference"
RMS_NORM_CASE_PATH = "_rms_norm_kernel/test__rms_norm_kernel_1.py"
RMS_NORM_CASE_SOURCE_SHA256 = "ce7fcc9e2882d0bb84c5fed0fc61bef56ea483bb59f1f138bebb89ae15d81872"
SELECTIVE_SCAN_CASE_PATH = (
    "_selective_scan_update_kernel/test__selective_scan_update_kernel_1.py"
)
SELECTIVE_SCAN_CASE_SOURCE_SHA256 = (
    "af063acf57be39ed697418cb8dac26c408a7915840893845c77c9472c62a3151"
)


@dataclass(frozen=True)
class PublicCaseRecord:
    """One visible test script, without claiming it is an executable EvaluationCase."""

    operator_name: str
    case_id: str
    source_path: str
    source_sha256: str
    identity_signature: str
    identity_signature_kind: str = IDENTITY_SIGNATURE_KIND
    evaluation_case_signatures: Tuple[str, ...] = ()
    materialization_status: str = "unmaterialized"
    materialization_reasons: Tuple[str, ...] = ("missing_explicit_evaluation_contract",)

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_case_signatures, tuple):
            raise ValueError("evaluation_case_signatures must be a tuple")
        if any(
            not isinstance(signature, str) or not signature.strip()
            for signature in self.evaluation_case_signatures
        ):
            raise ValueError("evaluation case signatures must be non-empty strings")
        if len(self.evaluation_case_signatures) != len(set(self.evaluation_case_signatures)):
            raise ValueError("evaluation case signatures must be unique")
        object.__setattr__(
            self, "evaluation_case_signatures", tuple(sorted(self.evaluation_case_signatures))
        )

    @property
    def evaluation_case_signature(self) -> str | None:
        """Compatibility view for records with exactly one materialized case."""
        if len(self.evaluation_case_signatures) == 1:
            return self.evaluation_case_signatures[0]
        return None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evaluation_case_signatures"] = list(self.evaluation_case_signatures)
        data["materialization_reasons"] = list(self.materialization_reasons)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PublicCaseRecord":
        signatures = data.get("evaluation_case_signatures")
        if signatures is None:
            legacy_signature = data.get("evaluation_case_signature")
            signatures = [] if legacy_signature is None else [legacy_signature]
        if not isinstance(signatures, list):
            raise ValueError("evaluation_case_signatures must be a list")
        return cls(
            operator_name=data["operator_name"],
            case_id=data["case_id"],
            source_path=data["source_path"],
            source_sha256=data["source_sha256"],
            identity_signature=data["identity_signature"],
            identity_signature_kind=data.get(
                "identity_signature_kind", IDENTITY_SIGNATURE_KIND
            ),
            evaluation_case_signatures=tuple(signatures),
            materialization_status=data.get("materialization_status", "unmaterialized"),
            materialization_reasons=tuple(data.get("materialization_reasons", [])),
        )


@dataclass(frozen=True)
class PublicCaseCatalog:
    """Deterministic inventory; only explicit EvaluationCase signatures are gate-ready."""

    dataset_root: str
    records: Tuple[PublicCaseRecord, ...]
    evidence_scope: str = "currently_visible_test_scripts"
    schema_version: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "records": [record.to_dict() for record in self.records],
            "evidence_scope": self.evidence_scope,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PublicCaseCatalog":
        schema_version = data.get("schema_version", 1)
        if schema_version not in {1, 2}:
            raise ValueError(f"unsupported public case catalog schema: {schema_version}")
        return cls(
            dataset_root=data["dataset_root"],
            records=tuple(PublicCaseRecord.from_dict(item) for item in data["records"]),
            evidence_scope=data.get("evidence_scope", "currently_visible_test_scripts"),
            schema_version=2,
        )

    def records_for(self, operator_name: str) -> Tuple[PublicCaseRecord, ...]:
        return tuple(record for record in self.records if record.operator_name == operator_name)

    def expected_case_signatures_for(self, operator_name: str) -> Tuple[str, ...]:
        records = self.records_for(operator_name)
        if not records:
            raise KeyError(f"operator not present in public case catalog: {operator_name}")
        unavailable = [record.case_id for record in records if not record.evaluation_case_signatures]
        if unavailable:
            raise ValueError(
                "gate-ready EvaluationCase signatures unavailable for: "
                + ", ".join(unavailable)
            )
        return tuple(
            sorted(
                signature
                for record in records
                for signature in record.evaluation_case_signatures
            )
        )


def build_public_case_catalog(dataset_root: Path) -> PublicCaseCatalog:
    """Catalog each visible test file once, preserving its exact source evidence."""

    root = dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    records: List[PublicCaseRecord] = []
    for operator_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for test_path in sorted(operator_dir.glob("test*.py")):
            source = test_path.read_bytes()
            ast.parse(source.decode("utf-8"), filename=str(test_path))
            relative_path = test_path.relative_to(root).as_posix()
            source_sha256 = hashlib.sha256(source).hexdigest()
            identity_payload = {
                "kind": IDENTITY_SIGNATURE_KIND,
                "operator_name": operator_dir.name,
                "source_path": relative_path,
                "source_sha256": source_sha256,
            }
            identity_signature = hashlib.sha256(
                json.dumps(
                    identity_payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            record = PublicCaseRecord(
                operator_name=operator_dir.name,
                case_id=relative_path,
                source_path=relative_path,
                source_sha256=source_sha256,
                identity_signature=identity_signature,
            )
            if relative_path == ACT_QUANT_CASE_PATH:
                if source_sha256 == ACT_QUANT_CASE_SOURCE_SHA256:
                    cases = materialize_act_quant_public_cases(root)
                    record = replace(
                        record,
                        evaluation_case_signatures=tuple(
                            case.signature() for case in cases
                        ),
                        materialization_status="materialized_explicit_manifest",
                        materialization_reasons=(),
                    )
                else:
                    record = replace(
                        record,
                        materialization_reasons=("source_sha256_mismatch",),
                    )
            elif relative_path == CHUNK_CUMSUM_CASE_PATH:
                if source_sha256 == CHUNK_CUMSUM_CASE_SOURCE_SHA256:
                    case = materialize_chunk_cumsum_public_case(root)
                    record = replace(
                        record,
                        evaluation_case_signatures=(case.signature(),),
                        materialization_status="materialized_explicit_manifest",
                        materialization_reasons=(),
                    )
                else:
                    record = replace(
                        record,
                        materialization_reasons=("source_sha256_mismatch",),
                    )
            elif relative_path == COUNT_EXPERT_CASE_PATH:
                if source_sha256 == COUNT_EXPERT_CASE_SOURCE_SHA256:
                    case = materialize_count_expert_basic_public_case(root)
                    record = replace(
                        record,
                        evaluation_case_signatures=(case.signature(),),
                        materialization_status="materialized_explicit_manifest",
                        materialization_reasons=(),
                    )
                else:
                    record = replace(
                        record,
                        materialization_reasons=("source_sha256_mismatch",),
                    )
            elif relative_path == QUANTIZE_K_CACHE_CASE_PATH:
                if source_sha256 == QUANTIZE_K_CACHE_CASE_SOURCE_SHA256:
                    cases = materialize_quantize_k_cache_public_cases(root)
                    record = replace(
                        record,
                        evaluation_case_signatures=tuple(
                            case.signature() for case in cases
                        ),
                        materialization_status="materialized_explicit_manifest",
                        materialization_reasons=(),
                    )
                else:
                    record = replace(
                        record,
                        materialization_reasons=("source_sha256_mismatch",),
                    )
            elif relative_path == SET_K_AND_S_CASE_PATH:
                if source_sha256 == SET_K_AND_S_CASE_SOURCE_SHA256:
                    case = materialize_set_k_and_s_public_case(root)
                    record = replace(
                        record,
                        evaluation_case_signatures=(case.signature(),),
                        materialization_status="materialized_explicit_manifest",
                        materialization_reasons=(),
                    )
                else:
                    record = replace(
                        record,
                        materialization_reasons=("source_sha256_mismatch",),
                    )
            elif relative_path == PER_GROUP_TRANSPOSE_CASE_PATH:
                if source_sha256 == PER_GROUP_TRANSPOSE_CASE_SOURCE_SHA256:
                    case = materialize_per_group_transpose_public_case(root)
                    record = replace(
                        record,
                        evaluation_case_signatures=(case.signature(),),
                        materialization_status="materialized_explicit_manifest",
                        materialization_reasons=(),
                    )
                else:
                    record = replace(
                        record,
                        materialization_reasons=("source_sha256_mismatch",),
                    )
            elif relative_path == PACK_SEQ_CASE_PATH:
                if source_sha256 == PACK_SEQ_CASE_SOURCE_SHA256:
                    case = materialize_pack_seq_public_case(root)
                    record = replace(
                        record,
                        evaluation_case_signatures=(case.signature(),),
                        materialization_status="materialized_explicit_manifest",
                        materialization_reasons=(),
                    )
                else:
                    record = replace(
                        record,
                        materialization_reasons=("source_sha256_mismatch",),
                    )
            elif relative_path == RMS_NORM_CASE_PATH:
                if source_sha256 == RMS_NORM_CASE_SOURCE_SHA256:
                    case = materialize_rms_norm_public_case(root)
                    record = replace(
                        record,
                        evaluation_case_signatures=(case.signature(),),
                        materialization_status="materialized_explicit_manifest",
                        materialization_reasons=(),
                    )
                else:
                    record = replace(
                        record,
                        materialization_reasons=("source_sha256_mismatch",),
                    )
            elif relative_path == SELECTIVE_SCAN_CASE_PATH:
                if source_sha256 == SELECTIVE_SCAN_CASE_SOURCE_SHA256:
                    case = materialize_selective_scan_public_case(root)
                    record = replace(
                        record,
                        evaluation_case_signatures=(case.signature(),),
                        materialization_status="materialized_explicit_manifest",
                        materialization_reasons=(),
                    )
                else:
                    record = replace(
                        record,
                        materialization_reasons=("source_sha256_mismatch",),
                    )
            records.append(record)

    return PublicCaseCatalog(dataset_root=root.as_posix(), records=tuple(records))


def materialize_act_quant_public_cases(
    dataset_root: Path,
) -> Tuple[EvaluationCase, EvaluationCase]:
    """Materialize the default-value and round-shape public invocations."""

    root = dataset_root.resolve()
    test_path = root / ACT_QUANT_CASE_PATH
    source_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if source_sha256 != ACT_QUANT_CASE_SOURCE_SHA256:
        raise ValueError(
            "act-quant public case source SHA-256 mismatch; explicit manifest is stale"
        )

    def inputs(scale_fmt: str | None) -> InputContract:
        scalars: Dict[str, Any] = {"block_size": 128}
        if scale_fmt is not None:
            scalars["scale_fmt"] = scale_fmt
        return InputContract(
            tensors={
                "x": TensorInputContract(
                    shape=[2, 4, 256],
                    dtype="torch.float32",
                    layout="contiguous",
                    initializer=TensorInitializer("randn"),
                    mutable=False,
                )
            },
            scalars=scalars,
            alias_groups=[],
        )

    def execution(include_scale_fmt: bool) -> ExecutionBinding:
        arguments = [
            ArgumentBinding("x", "tensor", "x"),
            ArgumentBinding("block_size", "scalar", "block_size"),
        ]
        if include_scale_fmt:
            arguments.append(ArgumentBinding("scale_fmt", "scalar", "scale_fmt"))
        return ExecutionBinding(entrypoint="act_quant", arguments=arguments)

    def targets() -> List[OracleTarget]:
        return [
            OracleTarget(
                target_name="y",
                kind="output",
                candidate=ValueSelector("return", path=[0]),
                reference=ValueSelector("return", path=[0]),
                evidence="public_assertion",
            ),
            OracleTarget(
                target_name="scale",
                kind="output",
                candidate=ValueSelector("return", path=[1]),
                reference=ValueSelector("return", path=[1]),
                evidence="public_assertion",
            ),
        ]

    source_ref = f"{ACT_QUANT_CASE_PATH}@sha256:{ACT_QUANT_CASE_SOURCE_SHA256}"
    # The script does not seed RNG; zero is the local replay manifest choice.
    default_case = EvaluationCase(
        op_name="_act_quant_kernel",
        case_id=f"{ACT_QUANT_CASE_PATH}::default",
        inputs=inputs(None),
        seed=0,
        execution=execution(False),
        oracle_policy=OraclePolicy(
            reference_id=ACT_QUANT_DEFAULT_REFERENCE_ID,
            policy_id="act-quant-default-public-v1",
            kind="allclose",
            rtol=1e-2,
            atol=1e-2,
            equal_nan=False,
        ),
        oracle_targets=targets(),
        source="explicit_public_case_manifest",
        source_ref=f"{source_ref}#default",
    )
    round_case = EvaluationCase(
        op_name="_act_quant_kernel",
        case_id=f"{ACT_QUANT_CASE_PATH}::round",
        inputs=inputs("round"),
        seed=0,
        execution=execution(True),
        oracle_policy=OraclePolicy(
            reference_id=ACT_QUANT_ROUND_REFERENCE_ID,
            policy_id="act-quant-round-shape-public-v1",
            kind="shape",
        ),
        oracle_targets=targets(),
        source="explicit_public_case_manifest",
        source_ref=f"{source_ref}#round",
    )
    return default_case, round_case


def materialize_chunk_cumsum_public_case(dataset_root: Path) -> EvaluationCase:
    """Materialize the fixed invocation and two assertions in the public script."""

    test_path = dataset_root.resolve() / CHUNK_CUMSUM_CASE_PATH
    if hashlib.sha256(test_path.read_bytes()).hexdigest() != CHUNK_CUMSUM_CASE_SOURCE_SHA256:
        raise ValueError(
            "chunk-cumsum public case source SHA-256 mismatch; manifest is stale"
        )

    def tensor(shape: List[int]) -> TensorInputContract:
        return TensorInputContract(
            shape=shape,
            dtype="torch.float32",
            layout="contiguous",
            initializer=TensorInitializer("randn"),
        )

    return EvaluationCase(
        op_name="_chunk_cumsum_fwd_kernel",
        case_id=CHUNK_CUMSUM_CASE_PATH,
        inputs=InputContract(
            tensors={
                "dt": tensor([2, 16, 4]),
                "A": tensor([4]),
                "dt_bias": tensor([4]),
            },
            scalars={
                "chunk_size": 8,
                "dt_softplus": True,
                "dt_limit": [0.0, 10.0],
            },
        ),
        # The public script is unseeded; zero is the explicit local replay choice.
        seed=0,
        execution=ExecutionBinding(
            entrypoint="_chunk_cumsum_fwd",
            arguments=[
                ArgumentBinding("dt", "tensor", "dt"),
                ArgumentBinding("A", "tensor", "A"),
                ArgumentBinding("chunk_size", "scalar", "chunk_size"),
                ArgumentBinding("dt_bias", "tensor", "dt_bias"),
                ArgumentBinding("dt_softplus", "scalar", "dt_softplus"),
                ArgumentBinding("dt_limit", "scalar", "dt_limit"),
            ],
        ),
        oracle_policy=OraclePolicy(
            reference_id=CHUNK_CUMSUM_REFERENCE_ID,
            policy_id="chunk-cumsum-fp32-public-v1",
            kind="allclose",
            rtol=1e-4,
            atol=1e-4,
        ),
        oracle_targets=[
            OracleTarget(
                "dA_cumsum",
                "output",
                ValueSelector("return", path=[0]),
                ValueSelector("return", path=[0]),
                "public_assertion",
            ),
            OracleTarget(
                "dt_out",
                "output",
                ValueSelector("return", path=[1]),
                ValueSelector("return", path=[1]),
                "public_assertion",
            ),
        ],
        source="explicit_public_case_manifest",
        source_ref=(
            f"{CHUNK_CUMSUM_CASE_PATH}@sha256:{CHUNK_CUMSUM_CASE_SOURCE_SHA256}"
        ),
    )


def materialize_count_expert_basic_public_case(dataset_root: Path) -> EvaluationCase:
    """Materialize only the fixed basic/no-map invocation in the public script."""

    root = dataset_root.resolve()
    test_path = root / COUNT_EXPERT_CASE_PATH
    source_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if source_sha256 != COUNT_EXPERT_CASE_SOURCE_SHA256:
        raise ValueError(
            "count-expert public case source SHA-256 mismatch; explicit manifest is stale"
        )

    return EvaluationCase(
        op_name="_count_expert_num_tokens",
        case_id=f"{COUNT_EXPERT_CASE_PATH}::basic-no-map",
        inputs=InputContract(
            tensors={
                "topk_ids": TensorInputContract(
                    shape=[8],
                    dtype="torch.int32",
                    layout="contiguous",
                    initializer=TensorInitializer(
                        "literal", {"values": [0, 1, 2, 0, 1, 3, 2, 0]}
                    ),
                    mutable=False,
                )
            },
            scalars={"num_local_experts": 4, "expert_map": None},
            alias_groups=[],
        ),
        seed=0,
        execution=ExecutionBinding(
            entrypoint="count_expert_num_tokens",
            arguments=[
                ArgumentBinding("topk_ids", "tensor", "topk_ids"),
                ArgumentBinding(
                    "num_local_experts", "scalar", "num_local_experts"
                ),
                ArgumentBinding("expert_map", "scalar", "expert_map"),
            ],
        ),
        oracle_policy=OraclePolicy(
            reference_id=COUNT_EXPERT_REFERENCE_ID,
            policy_id="count-expert-basic-no-map-public-v1",
            kind="exact",
        ),
        oracle_targets=[
            OracleTarget(
                target_name="expert_num_tokens",
                kind="output",
                candidate=ValueSelector("return"),
                reference=ValueSelector("return"),
                evidence="public_assertion",
            )
        ],
        source="explicit_public_case_manifest",
        source_ref=(
            f"{COUNT_EXPERT_CASE_PATH}@sha256:{COUNT_EXPERT_CASE_SOURCE_SHA256}"
            "#basic-no-map"
        ),
    )


def materialize_quantize_k_cache_public_cases(
    dataset_root: Path,
) -> Tuple[EvaluationCase, EvaluationCase]:
    """Materialize the public rope-value and output-metadata assertions."""

    root = dataset_root.resolve()
    test_path = root / QUANTIZE_K_CACHE_CASE_PATH
    source_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if source_sha256 != QUANTIZE_K_CACHE_CASE_SOURCE_SHA256:
        raise ValueError(
            "quantize-k-cache public case source SHA-256 mismatch; explicit manifest is stale"
        )

    def inputs() -> InputContract:
        return InputContract(
            tensors={
                "k_nope": TensorInputContract(
                    shape=[4, 512],
                    dtype="torch.bfloat16",
                    layout="contiguous",
                    initializer=TensorInitializer("randn"),
                    mutable=False,
                ),
                "k_rope": TensorInputContract(
                    shape=[4, 64],
                    dtype="torch.bfloat16",
                    layout="contiguous",
                    initializer=TensorInitializer("randn"),
                    mutable=False,
                ),
            },
            scalars={"group_size": 128},
            alias_groups=[],
        )

    def execution() -> ExecutionBinding:
        return ExecutionBinding(
            entrypoint="_quantize_k_cache_fast",
            arguments=[
                ArgumentBinding("k_nope", "tensor", "k_nope"),
                ArgumentBinding("k_rope", "tensor", "k_rope"),
                ArgumentBinding("group_size", "scalar", "group_size"),
            ],
        )

    source_ref = (
        f"{QUANTIZE_K_CACHE_CASE_PATH}@sha256:"
        f"{QUANTIZE_K_CACHE_CASE_SOURCE_SHA256}"
    )
    # The public script does not seed RNG; zero is the local replay manifest choice.
    rope_case = EvaluationCase(
        op_name="_quantize_k_cache_fast_kernel",
        case_id=f"{QUANTIZE_K_CACHE_CASE_PATH}::rope",
        inputs=inputs(),
        seed=0,
        execution=execution(),
        oracle_policy=OraclePolicy(
            reference_id=QUANTIZE_K_CACHE_ROPE_REFERENCE_ID,
            policy_id="quantize-k-cache-rope-bf16-public-v1",
            kind="allclose",
            rtol=1e-5,
            atol=1e-3,
            equal_nan=False,
        ),
        oracle_targets=[
            OracleTarget(
                target_name="rope",
                kind="output",
                candidate=ValueSelector(
                    "return", path=[{"tensor_slice": [1, 528, None]}]
                ),
                reference=ValueSelector("return"),
                evidence="public_assertion",
            )
        ],
        source="explicit_public_case_manifest",
        source_ref=f"{source_ref}#rope",
    )
    metadata_case = EvaluationCase(
        op_name="_quantize_k_cache_fast_kernel",
        case_id=f"{QUANTIZE_K_CACHE_CASE_PATH}::metadata",
        inputs=inputs(),
        seed=0,
        execution=execution(),
        oracle_policy=OraclePolicy(
            reference_id=QUANTIZE_K_CACHE_METADATA_REFERENCE_ID,
            policy_id="quantize-k-cache-output-metadata-public-v1",
            kind="metadata",
        ),
        oracle_targets=[
            OracleTarget(
                target_name="output",
                kind="output",
                candidate=ValueSelector("return"),
                reference=ValueSelector("return"),
                evidence="public_assertion",
            )
        ],
        source="explicit_public_case_manifest",
        source_ref=f"{source_ref}#metadata",
    )
    return rope_case, metadata_case


def materialize_set_k_and_s_public_case(dataset_root: Path) -> EvaluationCase:
    """Materialize the visible first-token K/scale assertions as raw bytes."""

    root = dataset_root.resolve()
    test_path = root / SET_K_AND_S_CASE_PATH
    source_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if source_sha256 != SET_K_AND_S_CASE_SOURCE_SHA256:
        raise ValueError(
            "set-k-and-s public case source SHA-256 mismatch; explicit manifest is stale"
        )

    return EvaluationCase(
        op_name="_set_k_and_s_triton_kernel",
        case_id=SET_K_AND_S_CASE_PATH,
        inputs=InputContract(
            tensors={
                "buf": TensorInputContract(
                    shape=[4, 64 * (128 + 4)],
                    dtype="torch.uint8",
                    layout="contiguous",
                    initializer=TensorInitializer("zeros"),
                    mutable=True,
                ),
                "loc": TensorInputContract(
                    shape=[3],
                    dtype="torch.int64",
                    layout="contiguous",
                    initializer=TensorInitializer(
                        "literal", {"values": [0, 64, 128]}
                    ),
                ),
                "index_k": TensorInputContract(
                    shape=[3, 128],
                    dtype="torch.float16",
                    layout="contiguous",
                    initializer=TensorInitializer("randn"),
                ),
                "index_k_scale": TensorInputContract(
                    shape=[3, 1],
                    dtype="torch.float32",
                    layout="contiguous",
                    initializer=TensorInitializer("randn"),
                ),
            },
            scalars={"page_size": 64},
            alias_groups=[],
        ),
        # The public script does not seed RNG; zero is the local replay declaration.
        seed=0,
        execution=ExecutionBinding(
            entrypoint="_set_k_and_s_triton",
            arguments=[
                ArgumentBinding("buf", "tensor", "buf"),
                ArgumentBinding("loc", "tensor", "loc"),
                ArgumentBinding("index_k", "tensor", "index_k"),
                ArgumentBinding("index_k_scale", "tensor", "index_k_scale"),
                ArgumentBinding("page_size", "scalar", "page_size"),
            ],
        ),
        oracle_policy=OraclePolicy(
            reference_id=SET_K_AND_S_REFERENCE_ID,
            policy_id="set-k-and-s-first-token-raw-bytes-manifest-v1",
            kind="exact",
        ),
        oracle_targets=[
            OracleTarget(
                target_name="k_bytes",
                kind="side_effect",
                candidate=ValueSelector(
                    "tensor",
                    tensor_name="buf",
                    path=[{"tensor_flat_slice": [0, 256]}],
                ),
                reference=ValueSelector("return", path=[0]),
                evidence="manifest_strengthening",
            ),
            OracleTarget(
                target_name="scale_bytes",
                kind="side_effect",
                candidate=ValueSelector(
                    "tensor",
                    tensor_name="buf",
                    path=[{"tensor_flat_slice": [8192, 8196]}],
                ),
                reference=ValueSelector("return", path=[1]),
                evidence="manifest_strengthening",
            ),
        ],
        source="explicit_public_case_manifest",
        source_ref=(
            f"{SET_K_AND_S_CASE_PATH}@sha256:{SET_K_AND_S_CASE_SOURCE_SHA256}"
            "#first-token-raw-byte-strengthening"
        ),
    )


def materialize_per_group_transpose_public_case(
    dataset_root: Path,
) -> EvaluationCase:
    """Materialize the fixed public input with an explicit semantic strengthening."""

    root = dataset_root.resolve()
    test_path = root / PER_GROUP_TRANSPOSE_CASE_PATH
    source_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if source_sha256 != PER_GROUP_TRANSPOSE_CASE_SOURCE_SHA256:
        raise ValueError(
            "per-group-transpose public case source SHA-256 mismatch; "
            "explicit manifest is stale"
        )

    return EvaluationCase(
        op_name="_per_group_transpose",
        case_id=PER_GROUP_TRANSPOSE_CASE_PATH,
        inputs=InputContract(
            tensors={
                "a": TensorInputContract(
                    shape=[32, 16],
                    dtype="torch.float32",
                    layout="contiguous",
                    initializer=TensorInitializer("randn"),
                ),
                "expert_offsets": TensorInputContract(
                    shape=[3],
                    dtype="torch.int32",
                    layout="contiguous",
                    initializer=TensorInitializer(
                        "literal", {"values": [0, 16, 32]}
                    ),
                ),
            },
            scalars={"M_ALIGNMENT": 1},
            alias_groups=[],
        ),
        # The public script does not seed RNG; zero is the local replay declaration.
        seed=0,
        execution=ExecutionBinding(
            entrypoint="per_group_transpose",
            arguments=[
                ArgumentBinding("a", "tensor", "a"),
                ArgumentBinding("expert_offsets", "tensor", "expert_offsets"),
                ArgumentBinding("M_ALIGNMENT", "scalar", "M_ALIGNMENT"),
            ],
        ),
        oracle_policy=OraclePolicy(
            reference_id=PER_GROUP_TRANSPOSE_REFERENCE_ID,
            policy_id="per-group-transpose-blockwise-fp32-manifest-v1",
            kind="exact",
        ),
        oracle_targets=[
            OracleTarget(
                target_name="output",
                kind="output",
                candidate=ValueSelector("return"),
                reference=ValueSelector("return"),
                evidence="manifest_strengthening",
            )
        ],
        source="explicit_public_case_manifest",
        source_ref=(
            f"{PER_GROUP_TRANSPOSE_CASE_PATH}@sha256:"
            f"{PER_GROUP_TRANSPOSE_CASE_SOURCE_SHA256}#blockwise-transpose-strengthening"
        ),
    )


def materialize_pack_seq_public_case(dataset_root: Path) -> EvaluationCase:
    """Materialize the one fixed invocation and assertion in the public script."""

    test_path = dataset_root.resolve() / PACK_SEQ_CASE_PATH
    if hashlib.sha256(test_path.read_bytes()).hexdigest() != PACK_SEQ_CASE_SOURCE_SHA256:
        raise ValueError("pack-seq public case source SHA-256 mismatch; manifest is stale")
    return EvaluationCase(
        op_name="_pack_seq_kernel",
        case_id=PACK_SEQ_CASE_PATH,
        inputs=InputContract(
            tensors={
                "x": TensorInputContract(
                    shape=[4096, 4],
                    dtype="torch.float32",
                    layout="contiguous",
                    initializer=TensorInitializer("randn"),
                ),
                "lengths": TensorInputContract(
                    shape=[3],
                    dtype="torch.int32",
                    layout="contiguous",
                    initializer=TensorInitializer("literal", {"values": [3, 4, 3]}),
                ),
            },
            scalars={"pad_value": 0.0, "block_t": 32, "block_d": 32},
        ),
        seed=0,
        execution=ExecutionBinding(
            entrypoint="pack_seq_triton",
            arguments=[
                ArgumentBinding("x", "tensor", "x"),
                ArgumentBinding("lengths", "tensor", "lengths"),
                ArgumentBinding("pad_value", "scalar", "pad_value"),
                ArgumentBinding("block_t", "scalar", "block_t"),
                ArgumentBinding("block_d", "scalar", "block_d"),
            ],
        ),
        oracle_policy=OraclePolicy(
            PACK_SEQ_REFERENCE_ID,
            "pack-seq-public-allclose-v1",
            "allclose",
            rtol=1e-5,
            atol=1e-5,
        ),
        oracle_targets=[
            OracleTarget(
                "output",
                "output",
                ValueSelector("return"),
                ValueSelector("return"),
                "public_assertion",
            )
        ],
        source="explicit_public_case_manifest",
        source_ref=(
            f"{PACK_SEQ_CASE_PATH}@sha256:{PACK_SEQ_CASE_SOURCE_SHA256}#public-allclose"
        ),
    )


def materialize_rms_norm_public_case(dataset_root: Path) -> EvaluationCase:
    """Materialize the one reviewed RMSNorm script from an explicit manifest."""

    root = dataset_root.resolve()
    test_path = root / RMS_NORM_CASE_PATH
    source_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if source_sha256 != RMS_NORM_CASE_SOURCE_SHA256:
        raise ValueError(
            "RMSNorm public case source SHA-256 mismatch; explicit manifest is stale"
        )

    return EvaluationCase(
        op_name="_rms_norm_kernel",
        case_id=RMS_NORM_CASE_PATH,
        inputs=InputContract(
            tensors={
                "input_tensor": TensorInputContract(
                    shape=[32, 128, 512],
                    dtype="torch.float32",
                    layout="contiguous",
                    initializer=TensorInitializer("randn"),
                    mutable=False,
                ),
                "weight_tensor": TensorInputContract(
                    shape=[512],
                    dtype="torch.float32",
                    layout="contiguous",
                    initializer=TensorInitializer("randn"),
                    mutable=False,
                ),
            },
            scalars={"eps": 1e-6},
            alias_groups=[],
        ),
        seed=0,
        execution=ExecutionBinding(
            entrypoint="rms_norm",
            arguments=[
                ArgumentBinding("input", "tensor", "input_tensor"),
                ArgumentBinding("weight", "tensor", "weight_tensor"),
                ArgumentBinding("eps", "scalar", "eps"),
            ],
        ),
        oracle_policy=OraclePolicy(
            reference_id=f"{RMS_NORM_CASE_PATH}::torch_output",
            policy_id="rms-norm-fp32-public-v1",
            kind="allclose",
            rtol=1e-4,
            atol=1e-5,
            equal_nan=False,
        ),
        oracle_targets=[
            OracleTarget(
                target_name="output",
                kind="output",
                candidate=ValueSelector("return"),
                reference=ValueSelector("return"),
                evidence="public_assertion",
            )
        ],
        source="explicit_public_case_manifest",
        source_ref=f"{RMS_NORM_CASE_PATH}@sha256:{RMS_NORM_CASE_SOURCE_SHA256}",
    )


def materialize_selective_scan_public_case(dataset_root: Path) -> EvaluationCase:
    """Materialize the reviewed stateful selective-scan public script."""

    root = dataset_root.resolve()
    test_path = root / SELECTIVE_SCAN_CASE_PATH
    source_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    if source_sha256 != SELECTIVE_SCAN_CASE_SOURCE_SHA256:
        raise ValueError(
            "selective-scan public case source SHA-256 mismatch; explicit manifest is stale"
        )

    def tensor(
        shape: List[int], *, initializer: str = "randn", mutable: bool = False
    ) -> TensorInputContract:
        return TensorInputContract(
            shape=shape,
            dtype="torch.float32",
            layout="contiguous",
            initializer=TensorInitializer(initializer),
            mutable=mutable,
        )

    return EvaluationCase(
        op_name="_selective_scan_update_kernel",
        case_id=SELECTIVE_SCAN_CASE_PATH,
        inputs=InputContract(
            tensors={
                "state": tensor([2, 4, 64, 16], mutable=True),
                "x": tensor([2, 4, 64]),
                "dt": tensor([2, 4, 64]),
                "A": tensor([4, 64, 16]),
                "B": tensor([2, 2, 16]),
                "C": tensor([2, 2, 16]),
                "D": tensor([4, 64]),
                "z": tensor([2, 4, 64]),
                "dt_bias": tensor([4, 64]),
                "out": tensor([2, 4, 64], initializer="zeros", mutable=True),
            },
            scalars={
                "dt_softplus": True,
                "state_batch_indices": None,
                "pad_slot_id": -1,
            },
            alias_groups=[],
        ),
        # The public script does not seed RNG; zero is an explicit manifest declaration.
        seed=0,
        execution=ExecutionBinding(
            entrypoint="selective_state_update",
            arguments=[
                ArgumentBinding("state", "tensor", "state"),
                ArgumentBinding("x", "tensor", "x"),
                ArgumentBinding("dt", "tensor", "dt"),
                ArgumentBinding("A", "tensor", "A"),
                ArgumentBinding("B", "tensor", "B"),
                ArgumentBinding("C", "tensor", "C"),
                ArgumentBinding("D", "tensor", "D"),
                ArgumentBinding("z", "tensor", "z"),
                ArgumentBinding("dt_bias", "tensor", "dt_bias"),
                ArgumentBinding("dt_softplus", "scalar", "dt_softplus"),
                ArgumentBinding(
                    "state_batch_indices", "scalar", "state_batch_indices"
                ),
                ArgumentBinding("pad_slot_id", "scalar", "pad_slot_id"),
                ArgumentBinding("out", "tensor", "out"),
            ],
        ),
        oracle_policy=OraclePolicy(
            # The public assertion compares out only; final state is not an observed oracle output.
            reference_id=f"{SELECTIVE_SCAN_CASE_PATH}::SelectiveScanUpdateReference",
            policy_id="selective-scan-out-state-fp32-manifest-v2",
            kind="allclose",
            rtol=1e-4,
            atol=1e-4,
            equal_nan=False,
        ),
        oracle_targets=[
            OracleTarget(
                target_name="out",
                kind="output",
                candidate=ValueSelector("tensor", tensor_name="out"),
                reference=ValueSelector("return", path=[0]),
                evidence="public_assertion",
            ),
            OracleTarget(
                target_name="state",
                kind="side_effect",
                candidate=ValueSelector("tensor", tensor_name="state"),
                reference=ValueSelector("return", path=[1]),
                evidence="manifest_strengthening",
            ),
        ],
        source="explicit_public_case_manifest",
        source_ref=(
            f"{SELECTIVE_SCAN_CASE_PATH}@sha256:{SELECTIVE_SCAN_CASE_SOURCE_SHA256}"
        ),
    )

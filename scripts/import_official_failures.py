#!/usr/bin/env python3
"""Import one completed CourseGrading run into official failure history."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wlz_optimizer.cache import OfficialFailureHistory
from wlz_optimizer.official_adapter import (
    bind_official_task_failures,
    parse_official_task_failures,
)


EXPECTED_TARGET = {
    "contest_id": "1mTsU6jaSZ0",
    "task_id": "14955089",
    "assignment_id": "47585",
    "problem_id": "3153461",
    "task_title": "2026年编译挑战赛-基于进化算法的Triton自动优化系统",
    "task_url": (
        "https://course.educg.net/pages/contest/contest_submit.jsp?"
        "contestID=1mTsU6jaSZ0&taskID=14955089&my=false&contestCID=0"
    ),
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _repo_file(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Official artifact path must be a non-empty string")
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("Official artifact path must remain inside the repository") from exc
    if not path.is_file():
        raise ValueError(f"Official artifact file does not exist: {path}")
    return path


def _manifest_for(archive: Path) -> Path:
    if archive.suffix != ".zip":
        raise ValueError("Official submission artifact must be a ZIP")
    return _repo_file(str(archive.with_suffix(".manifest.json")))


def _check_manifest_path(manifest: dict[str, Any], archive: Path) -> None:
    value = manifest.get("artifact_path")
    if not isinstance(value, str) or value != archive.name:
        raise ValueError("Submission manifest artifact path mismatch")


def _timestamp_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Saved run {field} must be a non-empty timestamp")
    timestamp = datetime.fromisoformat(value)
    if timestamp.utcoffset() is None:
        raise ValueError(f"Saved run {field} must include a timezone")
    return timestamp.strftime("%Y%m%d-%H%M%S")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    try:
        run_dir.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("Official run directory must remain inside the repository") from exc
    metadata = _load_object(run_dir / "metadata.json")
    target = metadata.get("target")
    if not isinstance(target, dict) or any(
        target.get(key) != value for key, value in EXPECTED_TARGET.items()
    ):
        raise ValueError("Saved run does not match the fixed CourseGrading task")
    levels = metadata.get("result_levels")
    if not isinstance(levels, dict) or not (
        levels.get("upload_accepted") is True
        and levels.get("platform_run_completed") is True
    ):
        raise ValueError("Saved run is not a completed accepted submission")

    observation_id = _timestamp_id(metadata.get("submission_time"), "submission_time")
    directory_ids = {observation_id}
    if metadata.get("completion_observed_at") is not None:
        directory_ids.add(
            _timestamp_id(metadata["completion_observed_at"], "completion_observed_at")
        )
    if not any(
        run_dir.name == value or run_dir.name.startswith(value + "-")
        for value in directory_ids
    ):
        raise ValueError(
            "Saved run directory does not match its submission or completion time"
        )
    environment = (
        f"coursegrading:contest={target['contest_id']}:task={target['task_id']}:"
        f"problem={target['problem_id']}:assign={target['assignment_id']}:"
        f"observation={observation_id}"
    )

    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Saved run has no artifact identity")
    archive = _repo_file(artifact.get("local_path"))
    archive_bytes = archive.read_bytes()
    if (
        artifact.get("filename") != archive.name
        or type(artifact.get("size_bytes")) is not int
        or artifact["size_bytes"] != len(archive_bytes)
    ):
        raise ValueError("Saved run artifact metadata mismatch")
    manifest = _load_object(_manifest_for(archive))
    _check_manifest_path(manifest, archive)
    base_manifest = None
    if manifest.get("base_artifact") is not None:
        base_link = manifest["base_artifact"]
        if not isinstance(base_link, dict):
            raise ValueError("Submission manifest has invalid base artifact link")
        base_archive = _repo_file(base_link.get("path"))
        if hashlib.sha256(base_archive.read_bytes()).hexdigest() != base_link.get("sha256"):
            raise ValueError("Submission base artifact SHA-256 mismatch")
        base_manifest = _load_object(_manifest_for(base_archive))
        _check_manifest_path(base_manifest, base_archive)

    failures = parse_official_task_failures(
        (run_dir / "raw-result.txt").read_text(encoding="utf-8")
    )
    official_result = metadata.get("official_result")
    if (
        not failures
        or not isinstance(official_result, dict)
        or type(official_result.get("failed_tasks")) is not int
        or official_result["failed_tasks"] != len(failures)
    ):
        raise ValueError("Saved run failure count mismatch")
    bound = bind_official_task_failures(
        failures,
        manifest,
        artifact.get("sha256"),
        artifact_bytes=archive_bytes,
        base_manifest=base_manifest,
    )

    history = OfficialFailureHistory(args.history)
    before = len(history)
    keys = history.append_many(bound, environment, observation_id)
    appended = len(history) - before
    print(json.dumps({
        "action": "imported",
        "appended": appended,
        "existing": len(bound) - appended,
        "failures": len(bound),
        "history_keys": keys,
        "observation_id": observation_id,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded, temporary-directory subprocess protocol for correctness workers."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, BinaryIO, Dict, Optional, Tuple


@dataclass(frozen=True)
class WorkerRequest:
    argv: Tuple[str, ...]
    timeout_seconds: float
    max_output_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if (
            not isinstance(self.argv, tuple)
            or not self.argv
            or any(not isinstance(item, str) or not item for item in self.argv)
        ):
            raise ValueError("worker argv must be a non-empty tuple of strings")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("worker timeout_seconds must be finite and positive")
        if (
            isinstance(self.max_output_bytes, bool)
            or not isinstance(self.max_output_bytes, int)
            or self.max_output_bytes < 0
        ):
            raise ValueError("worker max_output_bytes must be a non-negative integer")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "argv": list(self.argv),
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerRequest":
        if not isinstance(data.get("argv"), list):
            raise ValueError("serialized worker argv must be a list")
        return cls(
            argv=tuple(data["argv"]),
            timeout_seconds=data["timeout_seconds"],
            max_output_bytes=data.get("max_output_bytes", 64 * 1024),
        )


@dataclass(frozen=True)
class BoundedOutput:
    text: str
    total_bytes: int
    truncated: bool


@dataclass(frozen=True)
class WorkerResult:
    status: str
    returncode: Optional[int]
    stdout: BoundedOutput
    stderr: BoundedOutput
    duration_seconds: float
    working_directory: str

    def __post_init__(self) -> None:
        if self.status not in {"completed", "timeout", "launch_error"}:
            raise ValueError(f"unsupported worker status: {self.status}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _BoundedCollector(threading.Thread):
    def __init__(self, stream: BinaryIO, limit: int):
        super().__init__(daemon=True)
        self._stream = stream
        self._limit = limit
        self._kept = bytearray()
        self.total_bytes = 0

    def run(self) -> None:
        while True:
            chunk = self._stream.read(64 * 1024)
            if not chunk:
                return
            self.total_bytes += len(chunk)
            remaining = self._limit - len(self._kept)
            if remaining > 0:
                self._kept.extend(chunk[:remaining])

    def output(self) -> BoundedOutput:
        return BoundedOutput(
            text=bytes(self._kept).decode("utf-8", errors="replace"),
            total_bytes=self.total_bytes,
            truncated=self.total_bytes > len(self._kept),
        )


def run_worker(request: WorkerRequest) -> WorkerResult:
    """Run one command group in an ephemeral cwd with bounded in-memory logs."""

    if not isinstance(request, WorkerRequest):
        raise TypeError("request must be a WorkerRequest")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="wlz-correctness-worker-") as cwd:
        try:
            process = subprocess.Popen(
                request.argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            return WorkerResult(
                status="launch_error",
                returncode=None,
                stdout=BoundedOutput("", 0, False),
                stderr=_bounded_text(str(exc), request.max_output_bytes),
                duration_seconds=time.monotonic() - started,
                working_directory=cwd,
            )

        assert process.stdout is not None and process.stderr is not None
        stdout = _BoundedCollector(process.stdout, request.max_output_bytes)
        stderr = _BoundedCollector(process.stderr, request.max_output_bytes)
        stdout.start()
        stderr.start()

        timed_out = False
        try:
            process.wait(timeout=request.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(process)
            process.wait()
        finally:
            # Descendants may outlive a normally exited worker while retaining
            # its stdout/stderr pipes. Kill the isolated group before joining.
            _kill_process_tree(process)
            stdout.join()
            stderr.join()
            process.stdout.close()
            process.stderr.close()

        return WorkerResult(
            status="timeout" if timed_out else "completed",
            returncode=process.returncode,
            stdout=stdout.output(),
            stderr=stderr.output(),
            duration_seconds=time.monotonic() - started,
            working_directory=cwd,
        )


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        try:
            process.kill()
        except OSError:
            pass


def _bounded_text(text: str, limit: int) -> BoundedOutput:
    encoded = text.encode("utf-8")
    kept = encoded[:limit]
    return BoundedOutput(
        text=kept.decode("utf-8", errors="replace"),
        total_bytes=len(encoded),
        truncated=len(encoded) > len(kept),
    )

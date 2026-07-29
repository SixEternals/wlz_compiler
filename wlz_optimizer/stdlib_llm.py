"""Small OpenAI-compatible client used when optional LangChain deps are absent."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from wlz_optimizer.budget import BudgetController


REQUEST_FINGERPRINT_VERSION = "stdlib-openai-request-v1"
REQUEST_TIMEOUT_SECONDS = 120
SAFE_CALL_METADATA_KEYS = frozenset({
    "generation",
    "model",
    "mutation_type",
    "mutation_skill_version",
    "repair_skill_version",
    "mutation_plan_version",
    "mutation_plan_parent_sha256",
    "mutation_prompt_version",
    "operator_policy_version",
    "operator_policy_reason",
    "operator_policy_exploratory",
    "parent_fitness",
    "parents_fitness",
})


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class StdlibOpenAIClient:
    """Duck-typed subset of the organizer LLMInterface used by GeneticOperators."""

    requires_prompt_brace_escaping = False

    def __init__(self, config: Any, model_name: Optional[str] = None) -> None:
        self.config = config
        self.current_model = model_name or config.llm_models[0]
        self.api_url = (config.api_url or os.environ.get("API_URL", "")).rstrip("/")
        self._api_key = config.api_key or os.environ.get("API_KEY")
        if not self.api_url or not self._api_key:
            raise ValueError("API_URL and API_KEY must be configured")
        self.provider_sha256 = hashlib.sha256(
            self.api_url.encode("utf-8")
        ).hexdigest()
        self.budget_controller = getattr(config, "budget_controller", None)
        if self.budget_controller is not None and not isinstance(
            self.budget_controller, BudgetController
        ):
            raise TypeError("budget_controller must be a BudgetController or None")
        self.call_history: List[Dict[str, Any]] = []

    def switch_model(self, new_model: str) -> None:
        self.current_model = new_model

    def generate(
        self,
        prompt: str,
        system_msg: str = "",
        purpose: str = "unknown",
        **kwargs: Any,
    ) -> str:
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": prompt})
        serialized_messages = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload = {
            "model": self.current_model,
            "messages": messages,
            "temperature": self.config.llm_temperature,
            "max_tokens": self.config.max_llm_tokens,
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        metadata = {
            key: value for key, value in kwargs.items() if key in SAFE_CALL_METADATA_KEYS
        }
        system_prompt_version = kwargs.get("system_prompt_version")
        version_metadata = {
            key: value for key, value in metadata.items() if key.endswith("_version")
        }
        request_fingerprint = _canonical_sha256({
            "fingerprint_version": REQUEST_FINGERPRINT_VERSION,
            "messages": messages,
            "model": payload["model"],
            "temperature": payload["temperature"],
            "max_tokens": payload["max_tokens"],
            "thinking": payload["thinking"],
            "stream": payload["stream"],
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "provider_sha256": self.provider_sha256,
            "system_prompt_version": system_prompt_version,
            "version_metadata": version_metadata,
        })
        call_started = time.monotonic()
        call_record = {
            "purpose": purpose,
            "model": self.current_model,
            "prompt_sha256": hashlib.sha256(
                serialized_messages.encode("utf-8")
            ).hexdigest(),
            "request_fingerprint": request_fingerprint,
            "request_fingerprint_version": REQUEST_FINGERPRINT_VERSION,
            "system_prompt_version": system_prompt_version,
            "provider_sha256": self.provider_sha256,
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "retry_count": 0,
            "metadata": metadata,
        }

        def record_failure(error_type: str, *, http_status: Optional[int] = None) -> None:
            record = {
                **call_record,
                "status": "failed",
                "error_type": error_type,
                "latency_seconds": time.monotonic() - call_started,
            }
            if http_status is not None:
                record["http_status"] = http_status
            self.call_history.append(record)
        endpoint = (
            self.api_url
            if self.api_url.endswith("/chat/completions")
            else f"{self.api_url}/chat/completions"
        )
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        reservation = None
        if self.budget_controller is not None:
            decision = self.budget_controller.reserve(
                estimated_input_tokens=len(serialized_messages.encode("utf-8")),
                max_completion_tokens=self.config.max_llm_tokens,
                safety_margin_tokens=getattr(
                    self.config, "llm_token_safety_margin", 0
                ),
                expected_seconds=getattr(self.config, "llm_expected_seconds", 120),
            )
            if not decision.allowed:
                record_failure("budget_denied")
                raise RuntimeError(f"LLM budget denied request: {decision.reason}")
            reservation = decision.reservation
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # The server answered, so the call is not in an unknown in-flight
            # state; release the reservation instead of locking the budget.
            if reservation is not None:
                self.budget_controller.release(reservation)
            body = exc.read().decode("utf-8", errors="replace")
            record_failure("http_error", http_status=exc.code)
            raise RuntimeError(f"LLM HTTP {exc.code}: {body[:300]}") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if reservation is not None:
                self.budget_controller.mark_uncertain(reservation)
            record_failure(type(exc).__name__)
            raise RuntimeError(f"LLM request failed: {exc}") from None

        if not isinstance(result, dict):
            if reservation is not None:
                self.budget_controller.mark_uncertain(reservation)
            record_failure("invalid_json_type")
            raise RuntimeError("LLM response must be a JSON object")

        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        if reservation is not None:
            observed_tokens = usage.get("total_tokens")
            if (
                isinstance(observed_tokens, bool)
                or not isinstance(observed_tokens, int)
                or observed_tokens < 0
            ):
                observed_tokens = None
            self.budget_controller.commit(
                reservation,
                observed_tokens,
                fallback_tokens=reservation.token_upper_bound,
            )
        try:
            choice = result["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            record_failure("invalid_response_schema")
            raise RuntimeError("LLM response lacks choices[0].message.content") from exc
        finish_reason = choice.get("finish_reason")
        has_reasoning_content = bool(message.get("reasoning_content"))
        if not isinstance(content, str) or not content.strip():
            record_failure("empty_content")
            raise RuntimeError(
                "LLM returned empty content "
                f"(finish_reason={finish_reason}, "
                f"has_reasoning_content={has_reasoning_content})"
            )
        call_record.update({
            "status": "succeeded",
            "response_model": (
                result.get("model") if isinstance(result.get("model"), str) else None
            ),
            "response_id_sha256": (
                hashlib.sha256(result["id"].encode("utf-8")).hexdigest()
                if isinstance(result.get("id"), str)
                else None
            ),
            "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "usage": usage,
            "finish_reason": finish_reason,
            "has_reasoning_content": has_reasoning_content,
            "latency_seconds": time.monotonic() - call_started,
        })
        self.call_history.append(call_record)
        return content

    def get_stats(self) -> Dict[str, Any]:
        return {
            "call_count": len(self.call_history),
            "current_model": self.current_model,
            "calls": self.call_history,
        }

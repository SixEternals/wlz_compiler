#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Interface Module - Encapsulates Large Model Calls
"""

import os
import logging
import httpx
from typing import Dict, List, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import EAConfig


# All possible model list (for validation)
ALL_AVAILABLE_MODELS = [
    'deepseek-v4-pro', 'deepseek-v3.2', 'deepseek-r1',
    'qwen3.5', 'qwen3.6',
    'kimi-k2-instruct',
    'glm-4.6'
]


class LLMInterface:
    """
    Large Model Unified Interface - Supports Model Evolution

    Key features:
    1. Support model selection from list
    2. Model itself can be part of evolution
    """

    requires_prompt_brace_escaping = True

    def __init__(self, config: EAConfig, model_name: Optional[str] = None):
        """
        Initialize LLM interface

        Args:
            config: Configuration object
            model_name: Specify model name, if not specified use the first one in config
        """
        self.config = config
        self.budget_controller = getattr(config, "budget_controller", None)
        if self.budget_controller is not None and not all(
            callable(getattr(self.budget_controller, method, None))
            for method in ("snapshot", "reserve", "commit", "mark_uncertain")
        ):
            raise TypeError("budget_controller does not implement the budget protocol")
        self.total_tokens_used = (
            self.budget_controller.snapshot().used_tokens
            if self.budget_controller is not None
            else 0
        )
        self.budget_denial_reason: Optional[str] = None
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []

        # Select model: prioritize passed-in, otherwise use first in list
        if model_name:
            self.current_model = model_name
        elif config.llm_models:
            self.current_model = config.llm_models[0]
        else:
            raise ValueError("No available LLM, please add to config.llm_models")

        # Validate model
        if self.current_model not in ALL_AVAILABLE_MODELS:
            print(f"Warning: Model {self.current_model} is not in the standard list, but will still try to use it")

        self._init_llm()
        print(f"[LLM] Initialization complete: {self.current_model}")

    def _init_llm(self):
        """Initialize LLM client"""
        api_url = (self.config.api_url or os.getenv("API_URL", "https://api.siliconflow.cn/v1")).strip()
        api_key = self.config.api_key or os.getenv("API_KEY")

        if not api_key:
            raise ValueError("Please set API_KEY environment variable")

        http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
        https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
        proxy_url = https_proxy or http_proxy

        if proxy_url:
            print(f"[LLM] Using proxy: {proxy_url}")

        try:
            client_kwargs = {"timeout": 60.0, "verify": False}
            if proxy_url:
                client_kwargs["proxy"] = proxy_url
            client = httpx.Client(**client_kwargs)
        except TypeError:
            client_kwargs = {"timeout": 60.0, "verify": False}
            if proxy_url:
                client_kwargs["proxies"] = proxy_url
            client = httpx.Client(**client_kwargs)

        self.llm = ChatOpenAI(
            openai_api_base=api_url,
            openai_api_key=api_key,
            model=self.current_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_llm_tokens,
            http_client=client,
            openai_proxy=""
        )

    def switch_model(self, new_model: str):
        """
        Switch to a new model (for model evolution)

        Args:
            new_model: New model name
        """
        if new_model != self.current_model:
            print(f"[LLM] Switching model: {self.current_model} -> {new_model}")
            self.current_model = new_model
            self._init_llm()

    def generate(self, prompt: str, system_msg: str = "", 
                 purpose: str = "unknown", **kwargs) -> str:
        """
        Call LLM to generate text

        Args:
            prompt: User prompt
            system_msg: System prompt
            purpose: Call purpose (mutate/crossover, etc.)
            **kwargs: Extra information
        """
        self.call_count += 1

        purpose_types = {
            'mutate': 'Mutation', 'crossover': 'Crossover', 'analysis': 'Analysis',
            'fix': 'Fix', 'test_gen': 'Test Generation', 'optimize': 'Optimization',
            'model_evolve': 'Model Evolution', 'unknown': 'Unknown'
        }
        purpose_en = purpose_types.get(purpose, purpose)

        extra_info = []
        if 'parent_fitness' in kwargs:
            extra_info.append(f"parent_fitness={kwargs['parent_fitness']:.3f}")
        if 'parents_fitness' in kwargs:
            extra_info.append(f"parents_fitness={kwargs['parents_fitness']}")
        if 'mutation_type' in kwargs:
            extra_info.append(f"mutation_type={kwargs['mutation_type']}")
        if 'generation' in kwargs:
            extra_info.append(f"generation={kwargs['generation']}")
        if 'model' in kwargs:
            extra_info.append(f"model={kwargs['model']}")

        extra_str = f" ({', '.join(extra_info)})" if extra_info else ""

        print(f"[LLM] Call #{self.call_count:02d} | Model={self.current_model} | Purpose: {purpose_en}{extra_str}")

        messages = []
        if system_msg:
            messages.append(("system", system_msg))
        messages.append(("human", prompt))

        chat_prompt = ChatPromptTemplate.from_messages(messages)
        chain = chat_prompt | self.llm

        reservation = None
        if self.budget_controller is not None:
            decision = self.budget_controller.reserve(
                estimated_input_tokens=len(repr(messages).encode("utf-8")),
                max_completion_tokens=self.config.max_llm_tokens,
                safety_margin_tokens=getattr(
                    self.config, "llm_token_safety_margin", 0
                ),
                expected_seconds=getattr(self.config, "llm_expected_seconds", 120),
            )
            if not decision.allowed:
                self.budget_denial_reason = decision.reason or "budget_denied"
                error = f"LLM budget denied request: {decision.reason}"
                print(f"       └─ Failed: {error}")
                self.call_history.append({
                    'call_id': self.call_count,
                    'model': self.current_model,
                    'purpose': purpose,
                    'purpose_en': purpose_en,
                    'success': False,
                    'error': error,
                    'budget_denial_reason': self.budget_denial_reason,
                    'extra': kwargs
                })
                raise RuntimeError(error)
            self.budget_denial_reason = None
            reservation = decision.reservation

        try:
            response = chain.invoke({})
            generated_text = response.content

            usage = getattr(response, 'usage_metadata', None) or {}
            observed_tokens = usage.get('total_tokens')
            if (
                isinstance(observed_tokens, bool)
                or not isinstance(observed_tokens, int)
                or observed_tokens < 0
            ):
                observed_tokens = None
            cleaned_text = self._clean_code_markers(generated_text)
            if reservation is not None:
                tokens_used = (
                    observed_tokens
                    if observed_tokens is not None
                    else reservation.token_upper_bound
                )
                budget_snapshot = self.budget_controller.commit(
                    reservation,
                    observed_tokens,
                    fallback_tokens=reservation.token_upper_bound,
                )
                reservation = None
                self.total_tokens_used = budget_snapshot.used_tokens
            else:
                tokens_used = observed_tokens
                if tokens_used is None:
                    tokens_used = (
                        len(system_msg.split())
                        + len(prompt.split())
                        + len(generated_text.split())
                    )
                self.total_tokens_used += tokens_used

            print(f"       └─ Complete (~{tokens_used} tokens)")

            self.call_history.append({
                'call_id': self.call_count,
                'model': self.current_model,
                'purpose': purpose,
                'purpose_en': purpose_en,
                'tokens': tokens_used,
                'success': True,
                'extra': kwargs
            })

            return cleaned_text

        except Exception as e:
            if reservation is not None:
                budget_snapshot = self.budget_controller.mark_uncertain(reservation)
                self.total_tokens_used = budget_snapshot.used_tokens
            print(f"       └─ Failed: {str(e)}")
            self.call_history.append({
                'call_id': self.call_count,
                'model': self.current_model,
                'purpose': purpose,
                'purpose_en': purpose_en,
                'success': False,
                'error': str(e),
                'extra': kwargs
            })
            raise

    def _clean_code_markers(self, text: str) -> str:
        """Clean markdown markers"""
        import re
        text = re.sub(r'```python\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```\s*', '', text)
        return text.strip()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        purpose_stats = {}
        for call in self.call_history:
            p = call['purpose_en']
            if p not in purpose_stats:
                purpose_stats[p] = {'count': 0, 'tokens': 0, 'success': 0, 'models': set()}
            purpose_stats[p]['count'] += 1
            purpose_stats[p]['tokens'] += call.get('tokens', 0)
            purpose_stats[p]['models'].add(call.get('model', 'unknown'))
            if call['success']:
                purpose_stats[p]['success'] += 1

        # Convert set to list for JSON serialization
        for p in purpose_stats:
            purpose_stats[p]['models'] = list(purpose_stats[p]['models'])

        return {
            'total_tokens': self.total_tokens_used,
            'call_count': self.call_count,
            'current_model': self.current_model,
            'available_models': self.config.llm_models,
            'purpose_breakdown': purpose_stats
        }

    def print_call_summary(self):
        """Print call summary"""
        print(f"\n{'='*60}")
        print("LLM Call Statistics")
        print(f"{'='*60}")
        print(f"Current model: {self.current_model}")
        print(f"Available model list: {', '.join(self.config.llm_models)}")
        print(f"Total call count: {self.call_count}")
        print(f"Total token consumption: {self.total_tokens_used}")

        stats = self.get_stats()['purpose_breakdown']
        print(f"\nGrouped by purpose:")
        for purpose, data in sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True):
            success_rate = data['success']/data['count']*100 if data['count'] > 0 else 0
            models_str = ', '.join(data['models'])
            print(f"  - {purpose}: {data['count']} times, {data['tokens']} tokens, "
                  f"success rate {success_rate:.0f}%, models used: {models_str}")

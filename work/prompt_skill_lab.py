"""Generic prompt/Skill mechanism study translated from doge-code internals.

Reference commit: 2086e17ec5b05c7c2327ad55f9e18668e742a031
Relevant sources: systemPrompt.ts, SkillTool/prompt.ts, loadSkillsDir.ts.

Competition Skill content intentionally lives only in
``work/official_triton_agent/genetic_operators.py``.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


DOGE_CODE_COMMIT = "2086e17ec5b05c7c2327ad55f9e18668e742a031"
DEFAULT_SKILL_LISTING_CHAR_BUDGET = 8_000
MAX_LISTING_DESCRIPTION_CHARS = 250
MIN_LISTING_DESCRIPTION_CHARS = 20


def build_effective_system_prompt(
    default_sections: Sequence[str],
    *,
    custom_section: Optional[str] = None,
    append_section: Optional[str] = None,
    override_section: Optional[str] = None,
) -> tuple[str, ...]:
    """Preserve doge-code's override > custom > default, then append priority."""
    if override_section:
        return (override_section,)
    base = (custom_section,) if custom_section else tuple(default_sections)
    return base + ((append_section,) if append_section else ())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass(frozen=True)
class PromptSkill:
    name: str
    version: str
    description: str
    when_to_use: str
    content: str
    model_invocable: bool = True

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "description", "content"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if any(char.isspace() for char in self.name):
            raise ValueError("skill name must not contain whitespace")

    def listing_description(self, limit: int) -> str:
        text = (
            f"{self.description} - {self.when_to_use}"
            if self.when_to_use
            else self.description
        )
        return _truncate(text, limit)


@dataclass(frozen=True)
class ActivatedSkill:
    name: str
    version: str
    content: str


class SkillCatalog:
    """Expose bounded metadata first and full content only on exact activation."""

    def __init__(self, skills: Iterable[PromptSkill]) -> None:
        ordered = tuple(skills)
        if len({skill.name for skill in ordered}) != len(ordered):
            raise ValueError("skill names must be unique")
        self._ordered = ordered
        self._by_name = {skill.name: skill for skill in ordered}

    def discovery_listing(
        self,
        char_budget: int = DEFAULT_SKILL_LISTING_CHAR_BUDGET,
        max_description_chars: int = MAX_LISTING_DESCRIPTION_CHARS,
    ) -> str:
        visible = tuple(skill for skill in self._ordered if skill.model_invocable)
        if not visible:
            return ""
        if char_budget <= 0 or max_description_chars < 3:
            raise ValueError("skill listing budgets must be positive")
        entries = [
            f"- {skill.name}: {skill.listing_description(max_description_chars)}"
            for skill in visible
        ]
        full = "\n".join(entries)
        if len(full) <= char_budget:
            return full
        names_only = "\n".join(f"- {skill.name}" for skill in visible)
        if len(names_only) > char_budget:
            raise ValueError("skill names exceed the discovery budget")
        overhead = sum(len(f"- {skill.name}: ") for skill in visible) + len(visible) - 1
        per_description = (char_budget - overhead) // len(visible)
        if per_description < MIN_LISTING_DESCRIPTION_CHARS:
            return names_only
        return "\n".join(
            f"- {skill.name}: {skill.listing_description(per_description)}"
            for skill in visible
        )

    def activate(self, name: str) -> ActivatedSkill:
        skill = self._by_name.get(name)
        if skill is None or not skill.model_invocable:
            raise KeyError(f"unknown model-invocable skill: {name}")
        return ActivatedSkill(skill.name, skill.version, skill.content)

"""
Prompt Management

All LLM system prompts stored as .md files.
Easy to version, review, edit, and A/B test without touching Python code.

Structure:
- prompts/system/ - System prompts (identity, thinking, synthesis)
- prompts/extraction/ - Data extraction (facts, entities, summary)
- prompts/skills/ - Skill execution (code gen, SQL gen, analysis)
- prompts/meta/ - Meta/self-reflection (DMN, conflict resolution)
"""

from prompts.manager import PromptManager, prompts

__all__ = [
    "PromptManager",
    "prompts",
]

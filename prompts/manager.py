"""
Prompt Manager

Load and cache prompts from .md files.
Supports variable substitution with {{variable_name}} syntax.
"""

from pathlib import Path
from typing import Optional, Union


class PromptManager:
    """
    Load and cache prompts from .md files.
    
    Provides:
    - File-based prompt storage
    - Caching for performance
    - Variable substitution
    - Hot-reloading for development
    """
    
    def __init__(self, prompts_dir: Union[str, Path] = "prompts"):
        """
        Initialize prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt files.
        """
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, str] = {}
        self._identity_cache: Optional[str] = None
    
    def get(self, path: str, **variables) -> str:
        """
        Load a prompt by path.
        
        Supports variable substitution with {{variable_name}} syntax.
        
        Args:
            path: Path relative to prompts/ (e.g., "system/thinking.md").
            **variables: Variables to substitute in the prompt.
            
        Returns:
            Prompt text with variables substituted.
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist.
        """
        raw = self._load(path)
        return self._substitute(raw, variables)

    def get_messages(
        self,
        path: str,
        *,
        include_identity: bool = True,
        **variables,
    ) -> list[dict[str, str]]:
        """
        Load a prompt and split into system/user messages.

        The prompt file uses ``---`` on its own line to separate the
        system prompt (above) from the user prompt (below).
        If no separator is found, the entire file is treated as the
        system prompt with an empty user message.

        The agent identity (``system/identity.md``) is automatically
        prepended to the system message so the model always knows who
        it is.  Set ``include_identity=False`` to suppress.

        Variable substitution (``{{var}}``) is applied to both parts.

        Args:
            path: Path relative to prompts/.
            include_identity: Prepend identity prompt to system message.
            **variables: Variables to substitute.

        Returns:
            List of message dicts: [{"role": "system", ...}, {"role": "user", ...}].
        """
        raw = self._load(path)

        if "\n---\n" in raw:
            system_raw, user_raw = raw.split("\n---\n", 1)
        else:
            system_raw, user_raw = raw, ""

        system_content = self._substitute(system_raw, variables).strip()

        if include_identity:
            identity = self._get_identity(**variables)
            system_content = f"{identity}\n\n{system_content}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": self._substitute(user_raw, variables).strip()},
        ]

    # ── internals ─────────────────────────────────────────────────────────

    def _get_identity(self, **variables) -> str:
        """Load and cache the identity prompt."""
        if self._identity_cache is None:
            try:
                raw = self._load("system/identity.md")
                self._identity_cache = raw.strip()
            except FileNotFoundError:
                self._identity_cache = ""
        return self._substitute(self._identity_cache, variables)

    def _load(self, path: str) -> str:
        """Load and cache a prompt file."""
        if path not in self._cache:
            full_path = self.prompts_dir / path
            if not full_path.exists():
                raise FileNotFoundError(f"Prompt not found: {full_path}")
            self._cache[path] = full_path.read_text()
        return self._cache[path]

    def _substitute(self, text: str, variables: dict) -> str:
        """Apply {{variable}} substitution."""
        for key, value in variables.items():
            text = text.replace(f"{{{{{key}}}}}", str(value))
        return text
    
    def get_or_default(
        self,
        path: str,
        default: str = "",
        **variables,
    ) -> str:
        """
        Load a prompt, returning default if not found.
        
        Args:
            path: Path relative to prompts/.
            default: Default text if file not found.
            **variables: Variables to substitute.
            
        Returns:
            Prompt text or default.
        """
        try:
            return self.get(path, **variables)
        except FileNotFoundError:
            return default
    
    def exists(self, path: str) -> bool:
        """Check if a prompt file exists."""
        return (self.prompts_dir / path).exists()
    
    def reload(self, path: Optional[str] = None) -> None:
        """
        Clear cache.
        
        Useful for hot-reloading prompts during development.
        
        Args:
            path: Specific path to reload (None = clear all).
        """
        if path:
            self._cache.pop(path, None)
        else:
            self._cache.clear()
            self._identity_cache = None
    
    def list_prompts(self, subdir: str = "") -> list[str]:
        """
        List all prompt files in a subdirectory.
        
        Args:
            subdir: Subdirectory to list (e.g., "system").
            
        Returns:
            List of prompt paths relative to prompts/.
        """
        search_dir = self.prompts_dir / subdir if subdir else self.prompts_dir
        if not search_dir.exists():
            return []
        
        prompts = []
        for path in search_dir.rglob("*.md"):
            rel_path = path.relative_to(self.prompts_dir)
            prompts.append(str(rel_path))
        
        return sorted(prompts)


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

prompts = PromptManager()

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
        if path not in self._cache:
            full_path = self.prompts_dir / path
            if not full_path.exists():
                raise FileNotFoundError(f"Prompt not found: {full_path}")
            self._cache[path] = full_path.read_text()
        
        prompt = self._cache[path]
        
        # Variable substitution: {{variable_name}} → value
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
        
        return prompt
    
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

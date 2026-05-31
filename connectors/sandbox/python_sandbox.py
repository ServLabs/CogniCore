"""
Python Sandbox

Execute Python code in an isolated subprocess with restricted imports.
Uses AST validation to catch dangerous patterns before execution.
"""

import ast
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from core import config
from connectors.base import BaseSandbox, SandboxResult, ConnectorInfo, ConnectorStatus, ConnectorType


class PythonSandbox(BaseSandbox):
    """
    Python code sandbox.
    
    Provides:
    - AST validation (block dangerous imports/calls)
    - Subprocess isolation
    - Timeout enforcement
    - Output truncation
    
    Security policy from config.sandbox.
    """
    
    # Dangerous modules to block
    BLOCKED_IMPORTS = {
        "os", "sys", "subprocess", "shutil", "pathlib",
        "socket", "requests", "urllib", "http",
        "pickle", "marshal", "shelve",
        "ctypes", "multiprocessing", "threading",
        "__builtins__", "builtins",
    }
    
    # Dangerous function calls to block
    BLOCKED_CALLS = {
        "eval", "exec", "compile", "open", "input",
        "__import__", "getattr", "setattr", "delattr",
        "globals", "locals", "vars",
    }
    
    def __init__(self):
        """Initialize Python sandbox."""
        self._work_dir: Optional[Path] = None
    
    async def connect(self) -> None:
        """Set up sandbox working directory."""
        self._work_dir = config.paths.sandbox_tmp
        self._work_dir.mkdir(parents=True, exist_ok=True)
    
    async def disconnect(self) -> None:
        """Clean up sandbox."""
        pass
    
    async def health_check(self) -> bool:
        """Check if sandbox is working."""
        result = await self.execute("print('ok')")
        return result.success and result.output.strip() == "ok"
    
    async def execute(
        self,
        code: str,
        context: Optional[dict[str, Any]] = None,
    ) -> SandboxResult:
        """
        Execute Python code in sandbox.
        
        Args:
            code: Python code to execute.
            context: Variables to inject.
            
        Returns:
            SandboxResult with output or error.
        """
        # 1. Static validation
        violation = self._validate(code)
        if violation:
            return SandboxResult(
                success=False,
                output="",
                error=f"Security violation: {violation}",
                execution_time_ms=0,
                truncated=False,
            )
        
        # 2. Write to temp file
        script_path = self._work_dir / f"run_{int(time.time() * 1000)}.py"
        
        # 3. Inject context as variables
        preamble = ""
        if context:
            for key, value in context.items():
                preamble += f"{key} = {repr(value)}\n"
        
        script_path.write_text(preamble + code)
        
        # 4. Execute in subprocess with resource limits
        start = time.monotonic()
        try:
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=config.sandbox.python_timeout_seconds,
                cwd=str(self._work_dir),
                env={
                    "PATH": "/usr/bin:/usr/local/bin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            
            elapsed_ms = (time.monotonic() - start) * 1000
            
            # Truncate output if needed
            output = result.stdout
            truncated = False
            max_output = config.sandbox.max_output_bytes
            if len(output) > max_output:
                output = output[:max_output] + "\n... (truncated)"
                truncated = True
            
            if result.returncode == 0:
                return SandboxResult(
                    success=True,
                    output=output,
                    error=None,
                    execution_time_ms=elapsed_ms,
                    truncated=truncated,
                )
            else:
                return SandboxResult(
                    success=False,
                    output=output,
                    error=result.stderr[:max_output] if result.stderr else "Unknown error",
                    execution_time_ms=elapsed_ms,
                    truncated=truncated,
                )
        
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.monotonic() - start) * 1000
            return SandboxResult(
                success=False,
                output="",
                error=f"Execution timed out after {config.sandbox.python_timeout_seconds}s",
                execution_time_ms=elapsed_ms,
                truncated=False,
            )
        
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            return SandboxResult(
                success=False,
                output="",
                error=str(e),
                execution_time_ms=elapsed_ms,
                truncated=False,
            )
        
        finally:
            # Clean up script file
            try:
                script_path.unlink()
            except Exception:
                pass
    
    def _validate(self, code: str) -> Optional[str]:
        """
        Validate code using AST analysis.
        
        Args:
            code: Python code to validate.
            
        Returns:
            Violation message or None if valid.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in self.BLOCKED_IMPORTS:
                        return f"Blocked import: {module}"
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in self.BLOCKED_IMPORTS:
                        return f"Blocked import: {module}"
            
            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_CALLS:
                        return f"Blocked call: {node.func.id}"
        
        return None
    
    def info(self) -> ConnectorInfo:
        """Return connector info."""
        return ConnectorInfo(
            name="python_sandbox",
            connector_type=ConnectorType.SANDBOX,
            status=ConnectorStatus.CONNECTED if self._work_dir else ConnectorStatus.DISCONNECTED,
            metadata={
                "timeout_seconds": config.sandbox.python_timeout_seconds,
                "max_output_bytes": config.sandbox.max_output_bytes,
            },
        )

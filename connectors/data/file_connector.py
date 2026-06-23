"""
File Connector

Async file I/O: CSV, JSON, JSONL, Markdown, plain text.
All I/O is offloaded to threads via asyncio.to_thread.
"""

import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Optional, Union

from connectors.base import BaseConnector, ConnectorInfo, ConnectorStatus, ConnectorType


class FileConnector(BaseConnector):
    """
    Local file connector — async-native.
    
    Provides:
    - Read CSV, JSON, JSONL, Markdown, plain text
    - Write JSON, JSONL, text (append mode for JSONL)
    - Path resolution with optional base directory
    
    All I/O is async via asyncio.to_thread.
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir) if base_dir else None
    
    async def connect(self) -> None:
        """Ensure base directory exists."""
        if self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)
    
    async def disconnect(self) -> None:
        pass
    
    async def health_check(self) -> bool:
        return self.base_dir is None or self.base_dir.exists()
    
    async def read_csv(self, path: Union[str, Path]) -> list[dict[str, Any]]:
        """Read CSV file and return list of row dicts."""
        def _read():
            with open(self._resolve(path), newline="") as f:
                return list(csv.DictReader(f))
        return await asyncio.to_thread(_read)
    
    async def read_json(self, path: Union[str, Path]) -> Union[dict, list]:
        """Read JSON file."""
        def _read():
            with open(self._resolve(path)) as f:
                return json.load(f)
        return await asyncio.to_thread(_read)
    
    async def read_jsonl(self, path: Union[str, Path]) -> list[dict[str, Any]]:
        """Read JSONL file (one JSON object per line)."""
        def _read():
            with open(self._resolve(path)) as f:
                return [json.loads(line) for line in f if line.strip()]
        return await asyncio.to_thread(_read)
    
    async def read_text(self, path: Union[str, Path]) -> str:
        """Read plain text or markdown file."""
        def _read():
            with open(self._resolve(path)) as f:
                return f.read()
        return await asyncio.to_thread(_read)
    
    async def read_markdown(self, path: Union[str, Path]) -> str:
        """Read Markdown file."""
        return await self.read_text(path)
    
    async def write_json(
        self,
        path: Union[str, Path],
        data: Union[dict, list],
        indent: int = 2,
    ) -> None:
        """Write JSON file."""
        def _write():
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(data, f, indent=indent, default=str)
        await asyncio.to_thread(_write)
    
    async def write_jsonl(
        self,
        path: Union[str, Path],
        records: list[dict[str, Any]],
    ) -> None:
        """Append records to JSONL file."""
        def _write():
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                for record in records:
                    f.write(json.dumps(record, default=str) + "\n")
        await asyncio.to_thread(_write)
    
    async def write_text(
        self,
        path: Union[str, Path],
        content: str,
    ) -> None:
        """Write plain text file."""
        def _write():
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                f.write(content)
        await asyncio.to_thread(_write)
    
    async def exists(self, path: Union[str, Path]) -> bool:
        """Check if file exists."""
        return self._resolve(path).exists()
    
    async def list_dir(self, path: Union[str, Path]) -> list[str]:
        """List directory contents."""
        p = self._resolve(path)
        return [f.name for f in p.iterdir()] if p.is_dir() else []
    
    def _resolve(self, path: Union[str, Path]) -> Path:
        """Resolve path with optional base directory."""
        p = Path(path)
        if p.is_absolute():
            return p
        if self.base_dir:
            return self.base_dir / p
        return p
    
    def tool_schema(self) -> dict[str, Any]:
        """Expose file connector as a tool."""
        return {
            "name": "file_io",
            "description": (
                "Read or write local data files. Supports CSV, JSON, JSONL, "
                "Markdown, and plain text. Use for loading data, saving results, "
                "or checking file contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read_csv", "read_json", "read_jsonl", "read_text", "write_json", "write_text", "list_dir", "exists"],
                        "description": "File operation to perform.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path (relative to data directory).",
                    },
                    "data": {
                        "description": "Data to write (for write actions).",
                    },
                },
                "required": ["action", "path"],
            },
        }
    
    async def execute_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute file operation via tool interface."""
        action = params["action"]
        path = params["path"]
        
        try:
            match action:
                case "read_csv":
                    result = await self.read_csv(path)
                    return {"result": result}
                case "read_json":
                    result = await self.read_json(path)
                    return {"result": result}
                case "read_jsonl":
                    result = await self.read_jsonl(path)
                    return {"result": result}
                case "read_text":
                    result = await self.read_text(path)
                    return {"result": result}
                case "write_json":
                    await self.write_json(path, params.get("data", {}))
                    return {"result": f"Written to {path}"}
                case "write_text":
                    await self.write_text(path, params.get("data", ""))
                    return {"result": f"Written to {path}"}
                case "list_dir":
                    result = await self.list_dir(path)
                    return {"result": result}
                case "exists":
                    result = await self.exists(path)
                    return {"result": result}
                case _:
                    return {"error": f"Unknown action: {action}"}
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}
        except Exception as e:
            return {"error": str(e)}
    
    def info(self) -> ConnectorInfo:
        return ConnectorInfo(
            name="file",
            connector_type=ConnectorType.DATA,
            status=ConnectorStatus.CONNECTED,
            metadata={"base_dir": str(self.base_dir) if self.base_dir else None},
        )

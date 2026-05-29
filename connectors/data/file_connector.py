"""
File Connector

Read local files: CSV, JSON, JSONL, Markdown, plain text.
"""

import csv
import json
from pathlib import Path
from typing import Any, Optional, Union

from connectors.base import BaseConnector, ConnectorInfo, ConnectorStatus, ConnectorType


class FileConnector(BaseConnector):
    """
    Local file connector.
    
    Provides:
    - Read CSV, JSON, JSONL, Markdown, plain text
    - Write JSONL (append mode)
    - Path resolution with optional base directory
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize file connector.
        
        Args:
            base_dir: Base directory for relative paths.
        """
        self.base_dir = Path(base_dir) if base_dir else None
    
    async def connect(self) -> None:
        """No connection needed for local files."""
        pass
    
    async def disconnect(self) -> None:
        """No disconnection needed for local files."""
        pass
    
    async def health_check(self) -> bool:
        """Check if base directory exists (if configured)."""
        return self.base_dir is None or self.base_dir.exists()
    
    def read_csv(self, path: Union[str, Path]) -> list[dict[str, Any]]:
        """
        Read CSV file.
        
        Args:
            path: File path.
            
        Returns:
            List of row dicts.
        """
        with open(self._resolve(path), newline="") as f:
            return list(csv.DictReader(f))
    
    def read_json(self, path: Union[str, Path]) -> Union[dict, list]:
        """
        Read JSON file.
        
        Args:
            path: File path.
            
        Returns:
            Parsed JSON (dict or list).
        """
        with open(self._resolve(path)) as f:
            return json.load(f)
    
    def read_jsonl(self, path: Union[str, Path]) -> list[dict[str, Any]]:
        """
        Read JSONL file (one JSON object per line).
        
        Args:
            path: File path.
            
        Returns:
            List of parsed objects.
        """
        with open(self._resolve(path)) as f:
            return [json.loads(line) for line in f if line.strip()]
    
    def read_text(self, path: Union[str, Path]) -> str:
        """
        Read plain text file.
        
        Args:
            path: File path.
            
        Returns:
            File contents.
        """
        with open(self._resolve(path)) as f:
            return f.read()
    
    def read_markdown(self, path: Union[str, Path]) -> str:
        """
        Read Markdown file.
        
        Args:
            path: File path.
            
        Returns:
            File contents.
        """
        return self.read_text(path)
    
    def write_json(
        self,
        path: Union[str, Path],
        data: Union[dict, list],
        indent: int = 2,
    ) -> None:
        """
        Write JSON file.
        
        Args:
            path: File path.
            data: Data to write.
            indent: JSON indentation.
        """
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(data, f, indent=indent, default=str)
    
    def write_jsonl(
        self,
        path: Union[str, Path],
        records: list[dict[str, Any]],
    ) -> None:
        """
        Append records to JSONL file.
        
        Args:
            path: File path.
            records: Records to append.
        """
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
    
    def write_text(
        self,
        path: Union[str, Path],
        content: str,
    ) -> None:
        """
        Write plain text file.
        
        Args:
            path: File path.
            content: Content to write.
        """
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
    
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if file exists."""
        return self._resolve(path).exists()
    
    def list_dir(self, path: Union[str, Path]) -> list[str]:
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
    
    def info(self) -> ConnectorInfo:
        """Return connector info."""
        return ConnectorInfo(
            name="file",
            connector_type=ConnectorType.DATA,
            status=ConnectorStatus.CONNECTED,
            metadata={"base_dir": str(self.base_dir) if self.base_dir else None},
        )

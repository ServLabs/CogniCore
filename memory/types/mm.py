"""
Motor Memory (MM)

The agent's procedural knowledge — learned, reusable action sequences.
Analogous to human muscle memory: "I know how to do this."

This module provides:
- Procedure dataclass for executable/NL procedures
- MotorMemory manager with file + SQLite + FAISS storage
- Semantic procedure discovery via intent matching
- Usage tracking and success rate metrics

Note: Domain values come from config.py for platform-agnostic design.
"""

import importlib.util
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from collections.abc import Callable, Awaitable

import yaml
import numpy as np

from config import config


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Procedure:
    """
    A reusable procedure/skill.
    
    Can be executable Python code and/or natural language steps.
    """
    id: str
    name: str
    version: int
    domain: str  # From config.domains
    source: str  # "admin" or "system_generated"
    description: str
    intent_patterns: list[str] = field(default_factory=list)
    steps_nl: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    file_path_py: Optional[str] = None
    file_path_yaml: Optional[str] = None
    usage_count: int = 0
    success_count: int = 0
    last_used_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count
    
    @property
    def has_executable(self) -> bool:
        """Check if procedure has executable Python code."""
        return self.file_path_py is not None
    
    @property
    def has_nl_steps(self) -> bool:
        """Check if procedure has natural language steps."""
        return len(self.steps_nl) > 0
    
    @classmethod
    def create(
        cls,
        name: str,
        domain: str,
        description: str,
        source: str = "admin",
        intent_patterns: Optional[list[str]] = None,
        steps_nl: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> "Procedure":
        """Factory method to create a new procedure."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            version=1,
            domain=domain or config.domain_config.default_domain,
            source=source,
            description=description,
            intent_patterns=intent_patterns or [],
            steps_nl=steps_nl or [],
            tags=tags or [],
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Path, py_path: Optional[Path] = None) -> "Procedure":
        """Load procedure from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            version=data.get("version", 1),
            domain=data.get("domain", config.domain_config.default_domain),
            source=data.get("source", "admin"),
            description=data.get("description", ""),
            intent_patterns=data.get("intent_patterns", []),
            steps_nl=data.get("steps_nl", []),
            tags=data.get("tags", []),
            file_path_py=str(py_path) if py_path else None,
            file_path_yaml=str(yaml_path),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
        )
    
    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        data = {
            "name": self.name,
            "version": self.version,
            "domain": self.domain,
            "source": self.source,
            "description": self.description,
            "intent_patterns": self.intent_patterns,
            "steps_nl": self.steps_nl,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "domain": self.domain,
            "source": self.source,
            "description": self.description,
            "intent_patterns": self.intent_patterns,
            "steps_nl": self.steps_nl,
            "tags": self.tags,
            "file_path_py": self.file_path_py,
            "file_path_yaml": self.file_path_yaml,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def embedding_text(self) -> str:
        """Text to embed for semantic search."""
        parts = [self.name, self.description]
        parts.extend(self.intent_patterns)
        parts.extend(self.tags)
        return " ".join(parts)
    
    def to_prompt(self) -> str:
        """Generate prompt representation for LLM."""
        lines = [f"## Procedure: {self.name}"]
        lines.append(f"**Description**: {self.description}")
        
        if self.steps_nl:
            lines.append("\n**Steps**:")
            for i, step in enumerate(self.steps_nl, 1):
                lines.append(f"{i}. {step}")
        
        return "\n".join(lines)


@dataclass
class ProcedureResult:
    """Result of procedure execution."""
    procedure_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ══════════════════════════════════════════════════════════════════════════════
# Motor Memory Manager
# ══════════════════════════════════════════════════════════════════════════════

class MotorMemory:
    """
    Manager for procedural knowledge.
    
    Provides:
    - File-based procedure storage (human-editable)
    - SQLite metadata index for fast lookup
    - FAISS semantic search for intent matching
    - Usage tracking and success metrics
    
    Attributes:
        procedures_dir: Directory containing procedure files.
        db_path: Path to SQLite database.
        index_path: Path to FAISS index.
        embedder: Function to generate embeddings.
    """
    
    def __init__(
        self,
        procedures_dir: Optional[Path] = None,
        db_path: Optional[str] = None,
        index_path: Optional[Path] = None,
        embedder: Optional[Callable[[str], Awaitable[list[float]]]] = None,
        embedding_dim: int = 1024,
    ):
        """
        Initialize Motor Memory.
        
        Args:
            procedures_dir: Directory for procedure files.
            db_path: Path to SQLite database.
            index_path: Path to FAISS index.
            embedder: Async function to generate embeddings.
            embedding_dim: Dimension of embeddings.
        """
        self.procedures_dir = procedures_dir or config.paths.procedures_dir
        self.db_path = db_path or str(config.paths.hot_db)
        self.index_path = index_path or config.paths.faiss_dir / "mm.index"
        self.id_map_path = self.index_path.with_suffix(".idmap.json")
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        
        # Lazy initialization
        self._faiss_index = None
        self._id_map: dict[int, str] = {}  # FAISS ID → Procedure ID
        self._reverse_id_map: dict[str, int] = {}  # Procedure ID → FAISS ID
        self._next_faiss_id = 0
        
        # In-memory cache
        self._procedures: dict[str, Procedure] = {}
        self._by_name: dict[str, str] = {}  # name → id
        
        self._initialized = False
    
    # ── Initialization ──
    
    async def _init(self) -> None:
        """Initialize database and FAISS index."""
        if self._initialized:
            return
        
        await self._init_db()
        await self._init_faiss()
        await self._scan_procedures()
        self._initialized = True
    
    async def _init_db(self) -> None:
        """Initialize SQLite tables."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS procedures (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    version INTEGER DEFAULT 1,
                    domain TEXT,
                    source TEXT DEFAULT 'admin',
                    description TEXT,
                    file_path_py TEXT,
                    file_path_yaml TEXT,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    last_used_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_procedures_name 
                ON procedures(name)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_procedures_domain 
                ON procedures(domain)
            """)
            
            await db.commit()
    
    async def _init_faiss(self) -> None:
        """Initialize FAISS index."""
        import faiss
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.index_path.exists():
            self._faiss_index = faiss.read_index(str(self.index_path))
            
            if self.id_map_path.exists():
                data = json.loads(self.id_map_path.read_text())
                self._id_map = {int(k): v for k, v in data["id_map"].items()}
                self._reverse_id_map = {v: int(k) for k, v in data["id_map"].items()}
                self._next_faiss_id = data.get("next_id", len(self._id_map))
        else:
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
    
    async def _save_faiss(self) -> None:
        """Save FAISS index to disk."""
        import faiss
        
        faiss.write_index(self._faiss_index, str(self.index_path))
        
        data = {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "next_id": self._next_faiss_id,
        }
        self.id_map_path.write_text(json.dumps(data, indent=2))
    
    async def _scan_procedures(self) -> None:
        """Scan procedures directory and index new/updated procedures."""
        if not self.procedures_dir.exists():
            self.procedures_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Find all YAML files
        for yaml_path in self.procedures_dir.rglob("*.yaml"):
            py_path = yaml_path.with_suffix(".py")
            if not py_path.exists():
                py_path = None
            
            try:
                proc = Procedure.from_yaml(yaml_path, py_path)
                
                # Check if already indexed
                existing = await self._get_by_name(proc.name)
                if existing:
                    # Update if version changed
                    if proc.version > existing.version:
                        proc.id = existing.id
                        proc.usage_count = existing.usage_count
                        proc.success_count = existing.success_count
                        await self._save_procedure(proc)
                else:
                    await self._save_procedure(proc)
                    
            except Exception as e:
                # Log but don't fail on bad files
                print(f"Warning: Failed to load {yaml_path}: {e}")
    
    # ── Database Operations ──
    
    async def _save_procedure(self, proc: Procedure) -> None:
        """Save procedure to database and FAISS."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO procedures 
                (id, name, version, domain, source, description, 
                 file_path_py, file_path_yaml, usage_count, success_count,
                 last_used_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                proc.id, proc.name, proc.version, proc.domain, proc.source,
                proc.description, proc.file_path_py, proc.file_path_yaml,
                proc.usage_count, proc.success_count,
                proc.last_used_at.isoformat() if proc.last_used_at else None,
                proc.created_at.isoformat(), proc.updated_at.isoformat(),
            ))
            await db.commit()
        
        # Update FAISS
        if self.embedder and proc.id not in self._reverse_id_map:
            import faiss
            
            embedding = await self.embedder(proc.embedding_text())
            embedding_np = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(embedding_np)
            
            self._faiss_index.add(embedding_np)
            self._id_map[self._next_faiss_id] = proc.id
            self._reverse_id_map[proc.id] = self._next_faiss_id
            self._next_faiss_id += 1
            
            await self._save_faiss()
        
        # Update cache
        self._procedures[proc.id] = proc
        self._by_name[proc.name] = proc.id
    
    async def _get_by_name(self, name: str) -> Optional[Procedure]:
        """Get procedure by name from database."""
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM procedures WHERE name = ?",
                (name,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_procedure(dict(row))
        
        return None
    
    def _row_to_procedure(self, row: dict) -> Procedure:
        """Convert database row to Procedure."""
        # Load full data from YAML if available
        yaml_path = row.get("file_path_yaml")
        if yaml_path and Path(yaml_path).exists():
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f)
        else:
            yaml_data = {}
        
        return Procedure(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            domain=row["domain"] or config.domain_config.default_domain,
            source=row["source"] or "admin",
            description=row["description"] or "",
            intent_patterns=yaml_data.get("intent_patterns", []),
            steps_nl=yaml_data.get("steps_nl", []),
            tags=yaml_data.get("tags", []),
            file_path_py=row["file_path_py"],
            file_path_yaml=row["file_path_yaml"],
            usage_count=row["usage_count"] or 0,
            success_count=row["success_count"] or 0,
            last_used_at=datetime.fromisoformat(row["last_used_at"]) if row["last_used_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
    
    # ── Public API ──
    
    async def get_procedure(self, procedure_id: str) -> Optional[Procedure]:
        """Get procedure by ID."""
        await self._init()
        
        if procedure_id in self._procedures:
            return self._procedures[procedure_id]
        
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM procedures WHERE id = ?",
                (procedure_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    proc = self._row_to_procedure(dict(row))
                    self._procedures[proc.id] = proc
                    return proc
        
        return None
    
    async def find_by_name(self, name: str) -> Optional[Procedure]:
        """Find procedure by name."""
        await self._init()
        
        if name in self._by_name:
            return await self.get_procedure(self._by_name[name])
        
        return await self._get_by_name(name)
    
    async def find_by_intent(
        self,
        intent: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[Procedure]:
        """
        Find procedures matching an intent using semantic search.
        
        Args:
            intent: User intent string.
            top_k: Number of results.
            threshold: Minimum similarity score.
            
        Returns:
            List of matching procedures, ranked by relevance.
        """
        await self._init()
        
        if not self.embedder or self._faiss_index.ntotal == 0:
            return []
        
        import faiss
        
        embedding = await self.embedder(intent)
        embedding_np = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(embedding_np)
        
        scores, indices = self._faiss_index.search(embedding_np, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < threshold:
                continue
            
            proc_id = self._id_map.get(int(idx))
            if proc_id:
                proc = await self.get_procedure(proc_id)
                if proc:
                    results.append(proc)
        
        return results
    
    async def find_by_domain(
        self,
        domain: str,
        limit: int = 50,
    ) -> list[Procedure]:
        """Find procedures by domain."""
        await self._init()
        
        import aiosqlite
        
        procedures = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM procedures WHERE domain = ? ORDER BY usage_count DESC LIMIT ?",
                (domain, limit)
            ) as cursor:
                async for row in cursor:
                    procedures.append(self._row_to_procedure(dict(row)))
        
        return procedures
    
    async def execute(
        self,
        procedure: Procedure,
        context: Optional[dict[str, Any]] = None,
    ) -> ProcedureResult:
        """
        Execute a procedure.
        
        Args:
            procedure: The procedure to execute.
            context: Execution context (variables, etc.).
            
        Returns:
            ProcedureResult with output or error.
        """
        await self._init()
        
        start_time = datetime.now(timezone.utc)
        result = ProcedureResult(procedure_id=procedure.id, success=False)
        
        try:
            if procedure.has_executable and procedure.file_path_py:
                # Load and execute Python module
                spec = importlib.util.spec_from_file_location(
                    procedure.name,
                    procedure.file_path_py
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for execute() or run() function
                if hasattr(module, "execute"):
                    output = await module.execute(context or {})
                elif hasattr(module, "run"):
                    output = await module.run(context or {})
                else:
                    raise ValueError(f"Procedure {procedure.name} has no execute() or run() function")
                
                result.success = True
                result.output = output
            else:
                # NL-only procedure - return steps for LLM
                result.success = True
                result.output = {
                    "type": "nl_steps",
                    "steps": procedure.steps_nl,
                    "description": procedure.description,
                }
                
        except Exception as e:
            result.error = str(e)
        
        # Calculate duration
        end_time = datetime.now(timezone.utc)
        result.duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Update usage stats
        await self._record_usage(procedure, result.success)
        
        return result
    
    async def _record_usage(self, procedure: Procedure, success: bool) -> None:
        """Record procedure usage."""
        import aiosqlite
        
        procedure.usage_count += 1
        if success:
            procedure.success_count += 1
        procedure.last_used_at = datetime.now(timezone.utc)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE procedures 
                SET usage_count = ?, success_count = ?, last_used_at = ?
                WHERE id = ?
            """, (
                procedure.usage_count,
                procedure.success_count,
                procedure.last_used_at.isoformat(),
                procedure.id,
            ))
            await db.commit()
    
    async def create_procedure(
        self,
        name: str,
        domain: str,
        description: str,
        steps_nl: list[str],
        intent_patterns: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        source: str = "system_generated",
        code: Optional[str] = None,
    ) -> Procedure:
        """
        Create a new procedure (typically system-generated).
        
        Args:
            name: Procedure name.
            domain: Domain from config.
            description: Human-readable description.
            steps_nl: Natural language steps.
            intent_patterns: Patterns that match this procedure.
            tags: Tags for categorization.
            source: "admin" or "system_generated".
            code: Optional Python code.
            
        Returns:
            The created Procedure.
        """
        await self._init()
        
        proc = Procedure.create(
            name=name,
            domain=domain,
            description=description,
            source=source,
            intent_patterns=intent_patterns,
            steps_nl=steps_nl,
            tags=tags,
        )
        
        # Determine directory
        if source == "system_generated":
            proc_dir = self.procedures_dir / "_system_generated"
        else:
            proc_dir = self.procedures_dir / domain
        
        proc_dir.mkdir(parents=True, exist_ok=True)
        
        # Write YAML
        yaml_path = proc_dir / f"{name}.yaml"
        yaml_path.write_text(proc.to_yaml())
        proc.file_path_yaml = str(yaml_path)
        
        # Write Python if provided
        if code:
            py_path = proc_dir / f"{name}.py"
            py_path.write_text(code)
            proc.file_path_py = str(py_path)
        
        await self._save_procedure(proc)
        
        return proc
    
    async def list_procedures(
        self,
        domain: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> list[Procedure]:
        """List procedures with optional filters."""
        await self._init()
        
        import aiosqlite
        
        conditions = []
        params = []
        
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if source:
            conditions.append("source = ?")
            params.append(source)
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        procedures = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"SELECT * FROM procedures {where_clause} ORDER BY usage_count DESC LIMIT ?",
                params + [limit]
            ) as cursor:
                async for row in cursor:
                    procedures.append(self._row_to_procedure(dict(row)))
        
        return procedures


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_mm_instance: Optional[MotorMemory] = None


def get_motor_memory() -> MotorMemory:
    """
    Get the singleton MotorMemory instance.
    
    Returns:
        The MotorMemory instance.
    """
    global _mm_instance
    if _mm_instance is None:
        _mm_instance = MotorMemory()
    return _mm_instance


# ── Convenience Functions ──

async def find_procedure(intent: str) -> Optional[Procedure]:
    """Find a procedure matching the intent."""
    mm = get_motor_memory()
    results = await mm.find_by_intent(intent, top_k=1)
    return results[0] if results else None


async def execute_procedure(
    name: str,
    context: Optional[dict[str, Any]] = None,
) -> ProcedureResult:
    """Execute a procedure by name."""
    mm = get_motor_memory()
    proc = await mm.find_by_name(name)
    if not proc:
        return ProcedureResult(
            procedure_id="",
            success=False,
            error=f"Procedure not found: {name}",
        )
    return await mm.execute(proc, context)


async def learn_procedure(
    name: str,
    domain: str,
    description: str,
    steps: list[str],
) -> Procedure:
    """Create a system-generated procedure from learned steps."""
    return await get_motor_memory().create_procedure(
        name=name,
        domain=domain,
        description=description,
        steps_nl=steps,
        source="system_generated",
    )

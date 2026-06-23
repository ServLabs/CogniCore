"""
FAISS Index Migrations

Index versioning and rebuilding when embedding model changes.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import config
from logger import log
from observability import audit


# ══════════════════════════════════════════════════════════════════════════════
# Metadata Management
# ══════════════════════════════════════════════════════════════════════════════

def get_index_metadata(index_path: Path) -> dict[str, Any]:
    """
    Read index metadata (embedding model, dimension, count).
    
    Args:
        index_path: Path to FAISS index file.
        
    Returns:
        Metadata dict or empty dict if not found.
    """
    meta_path = index_path.with_suffix(".meta.json")
    
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, IOError):
            return {}
    
    return {}


def save_index_metadata(index_path: Path, metadata: dict[str, Any]) -> None:
    """
    Save index metadata.
    
    Args:
        index_path: Path to FAISS index file.
        metadata: Metadata to save.
    """
    meta_path = index_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# Compatibility Check
# ══════════════════════════════════════════════════════════════════════════════

async def check_index_compatibility(
    index_path: Path,
    embedder,
) -> bool:
    """
    Check if index is compatible with current embedding model.
    
    Args:
        index_path: Path to FAISS index file.
        embedder: Embedding connector.
        
    Returns:
        True if compatible, False if rebuild needed.
    """
    meta = get_index_metadata(index_path)
    
    if not meta:
        return False  # No metadata = needs rebuild
    
    # Check model name and dimension
    current_model = config.embedding.model_name
    current_dimension = embedder.dimension
    
    return (
        meta.get("model") == current_model and
        meta.get("dimension") == current_dimension
    )


# ══════════════════════════════════════════════════════════════════════════════
# Index Rebuilding
# ══════════════════════════════════════════════════════════════════════════════

async def rebuild_index(
    index_name: str,
    mml,
    embedder,
) -> int:
    """
    Rebuild a FAISS index with current embedding model.
    
    Args:
        index_name: Name of index (sfm, lfm, am, mm, meta).
        mml: Memory management layer.
        embedder: Embedding connector.
        
    Returns:
        Number of vectors indexed.
    """
    try:
        import faiss
        import numpy as np
    except ImportError:
        log.error("faiss-cpu not installed, cannot rebuild index")
        return 0
    
    audit.log_raw(
        "migration",
        "index_rebuild",
        "faiss",
        "started",
        target=index_name,
    )
    
    index_path = config.paths.faiss_dir / f"{index_name}.index"
    
    # Get all records that need re-embedding
    texts, ids = await _get_records_for_index(index_name, mml)
    
    if not texts:
        log.warning(f"No records found for index {index_name}")
        return 0
    
    log.info(f"Rebuilding {index_name} index with {len(texts)} records...")
    
    # Re-embed all
    embeddings = await embedder.embed_batch_async(texts)
    
    # Rebuild index
    dimension = embedder.dimension
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings).astype(np.float32))
    
    # Save index
    faiss.write_index(index, str(index_path))
    
    # Save ID map
    id_map = {i: ext_id for i, ext_id in enumerate(ids)}
    id_map_path = index_path.with_suffix(".idmap.json")
    id_map_path.write_text(json.dumps(id_map))
    
    # Save metadata
    save_index_metadata(index_path, {
        "model": config.embedding.model_name,
        "dimension": dimension,
        "count": len(ids),
        "rebuilt_at": datetime.now(timezone.utc).isoformat(),
    })
    
    audit.log_raw(
        "migration",
        "index_rebuild",
        "faiss",
        "completed",
        target=index_name,
        details={"count": len(ids)},
    )
    
    log.info(f"Index {index_name} rebuilt with {len(ids)} vectors")
    
    return len(ids)


async def _get_records_for_index(
    index_name: str,
    mml,
) -> tuple[list[str], list[str]]:
    """
    Get records that need to be indexed.
    
    Args:
        index_name: Name of index.
        mml: Memory management layer.
        
    Returns:
        Tuple of (texts, ids).
    """
    texts = []
    ids = []
    
    try:
        if index_name == "sfm":
            records = await mml.sfm_get_all()
            texts = [f"{r.subject} {r.predicate} {r.object}" for r in records]
            ids = [r.id for r in records]
        
        elif index_name == "lfm":
            chunks = await mml.lfm_get_all_chunks()
            texts = [c.text for c in chunks]
            ids = [c.id for c in chunks]
        
        elif index_name == "am":
            nodes = await mml.am_get_all_nodes()
            texts = [n.name for n in nodes]
            ids = [n.id for n in nodes]
        
        elif index_name == "mm":
            procedures = await mml.mm_get_all()
            texts = [p.name + " " + p.description for p in procedures]
            ids = [p.id for p in procedures]
        
        elif index_name == "meta":
            entries = await mml.meta_get_all()
            texts = [e.key for e in entries]
            ids = [e.id for e in entries]
    
    except Exception as e:
        log.error(f"Failed to get records for {index_name}: {e}")
    
    return texts, ids


async def rebuild_all_indexes(mml, embedder) -> dict[str, int]:
    """
    Rebuild all FAISS indexes.
    
    Args:
        mml: Memory management layer.
        embedder: Embedding connector.
        
    Returns:
        Dict mapping index name to vector count.
    """
    results = {}
    
    for index_name in ["sfm", "lfm", "am", "mm", "meta"]:
        count = await rebuild_index(index_name, mml, embedder)
        results[index_name] = count
    
    return results


async def check_all_indexes(embedder) -> dict[str, bool]:
    """
    Check compatibility of all indexes.
    
    Args:
        embedder: Embedding connector.
        
    Returns:
        Dict mapping index name to compatibility status.
    """
    results = {}
    
    for index_name in ["sfm", "lfm", "am", "mm", "meta"]:
        index_path = config.paths.faiss_dir / f"{index_name}.index"
        
        if index_path.exists():
            compatible = await check_index_compatibility(index_path, embedder)
            results[index_name] = compatible
        else:
            results[index_name] = True  # No index = no incompatibility
    
    return results

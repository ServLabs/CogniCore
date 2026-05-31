"""
Evaluations API Routes

REST endpoints for evaluation results and triggers:
- GET /evals - List recent eval results
- GET /evals/{eval_type} - Get results for specific eval type
- POST /evals/run - Trigger an eval run
- GET /evals/summary - Get eval summary
"""

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from audit import audit
from core import log
from evals import get_eval_runner, EvalResult


router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════

class EvalResultResponse(BaseModel):
    """Eval result response."""
    eval_type: str
    score: float
    sample_size: int
    passed: int
    failed: int
    details: dict[str, Any] = {}
    timestamp: str


class EvalSummary(BaseModel):
    """Summary of eval results."""
    eval_type: str
    count: int
    avg_score: float
    total_passed: int
    total_failed: int


class RunEvalRequest(BaseModel):
    """Request to run an eval."""
    eval_type: str
    params: dict[str, Any] = {}


class RunEvalResponse(BaseModel):
    """Response from running an eval."""
    eval_type: str
    status: str
    result: Optional[EvalResultResponse] = None
    error: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@router.get("", response_model=list[EvalResultResponse])
async def list_evals(
    eval_type: Optional[str] = Query(None, description="Filter by eval type"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
):
    """
    List recent eval results.
    
    Optionally filter by eval type.
    """
    runner = get_eval_runner()
    results = runner.get_recent_results(eval_type=eval_type, limit=limit)
    
    return [
        EvalResultResponse(
            eval_type=r.eval_type,
            score=r.score,
            sample_size=r.sample_size,
            passed=r.passed,
            failed=r.failed,
            details=r.details,
            timestamp=r.timestamp or datetime.now(timezone.utc).isoformat(),
        )
        for r in results
    ]


@router.get("/types")
async def list_eval_types():
    """
    List all registered eval types.
    """
    runner = get_eval_runner()
    return {
        "eval_types": runner.list_evals(),
    }


@router.get("/summary", response_model=list[EvalSummary])
async def get_eval_summary():
    """
    Get summary of all eval results.
    
    Returns aggregated stats per eval type.
    """
    runner = get_eval_runner()
    summary = runner.get_summary()
    
    return [
        EvalSummary(
            eval_type=eval_type,
            count=stats["count"],
            avg_score=stats.get("avg_score", 0.0),
            total_passed=stats["total_passed"],
            total_failed=stats["total_failed"],
        )
        for eval_type, stats in summary.items()
    ]


@router.get("/{eval_type}", response_model=list[EvalResultResponse])
async def get_eval_results(
    eval_type: str,
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Get results for a specific eval type.
    """
    runner = get_eval_runner()
    results = runner.get_recent_results(eval_type=eval_type, limit=limit)
    
    if not results:
        raise HTTPException(status_code=404, detail=f"No results for eval type: {eval_type}")
    
    return [
        EvalResultResponse(
            eval_type=r.eval_type,
            score=r.score,
            sample_size=r.sample_size,
            passed=r.passed,
            failed=r.failed,
            details=r.details,
            timestamp=r.timestamp or datetime.now(timezone.utc).isoformat(),
        )
        for r in results
    ]


@router.post("/run", response_model=RunEvalResponse)
async def run_eval(request: RunEvalRequest):
    """
    Trigger an eval run.
    
    Runs the specified eval type and returns the result.
    """
    runner = get_eval_runner()
    
    if request.eval_type not in runner.list_evals():
        raise HTTPException(status_code=404, detail=f"Unknown eval type: {request.eval_type}")
    
    audit.log_raw(
        "api",
        "run_eval",
        "evals",
        "started",
        target=request.eval_type,
    )
    
    try:
        result = await runner.run_single(request.eval_type)
        
        if result is None:
            return RunEvalResponse(
                eval_type=request.eval_type,
                status="failed",
                error="Eval returned no result",
            )
        
        audit.log_raw(
            "api",
            "run_eval",
            "evals",
            "completed",
            target=request.eval_type,
            details={"score": result.score},
        )
        
        return RunEvalResponse(
            eval_type=request.eval_type,
            status="completed",
            result=EvalResultResponse(
                eval_type=result.eval_type,
                score=result.score,
                sample_size=result.sample_size,
                passed=result.passed,
                failed=result.failed,
                details=result.details,
                timestamp=result.timestamp or datetime.now(timezone.utc).isoformat(),
            ),
        )
    
    except Exception as e:
        audit.log_raw(
            "api",
            "run_eval",
            "evals",
            "failed",
            target=request.eval_type,
            error=str(e),
        )
        
        return RunEvalResponse(
            eval_type=request.eval_type,
            status="failed",
            error=str(e),
        )


@router.post("/run-all")
async def run_all_evals():
    """
    Run all registered evals.
    
    Returns results for all evals.
    """
    runner = get_eval_runner()
    
    audit.log_raw("api", "run_all_evals", "evals", "started")
    
    results = await runner.run_all()
    
    audit.log_raw(
        "api",
        "run_all_evals",
        "evals",
        "completed",
        details={"count": len(results)},
    )
    
    return {
        "status": "completed",
        "results": [
            EvalResultResponse(
                eval_type=r.eval_type,
                score=r.score,
                sample_size=r.sample_size,
                passed=r.passed,
                failed=r.failed,
                details=r.details,
                timestamp=r.timestamp or datetime.now(timezone.utc).isoformat(),
            )
            for r in results
        ],
    }

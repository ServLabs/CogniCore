"""
Observational Learning (Learning Type 9)

Learn from watching admin/expert behavior. When an admin manually
performs a task, the agent observes the action trace and internalizes
it as a procedure.

Algorithm: Action trace capture + procedure extraction
Trigger: End of admin session (if trace length > threshold)
LLM: Yes (cheap - procedure generation from trace)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from logger import log
from observability import audit


@dataclass
class Action:
    """A single observed action."""
    action: str  # Action name/type
    params: dict[str, Any]  # Parameters used
    result: Optional[Any] = None  # Result if available
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "params": self.params,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ActionTrace:
    """A sequence of observed actions."""
    trace_id: str
    session_id: str
    user_id: str
    user_role: str  # "admin", "expert", "user"
    actions: list[Action]
    started_at: datetime
    ended_at: Optional[datetime] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_role": self.user_role,
            "actions": [a.to_dict() for a in self.actions],
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }


@dataclass
class ObservedProcedure:
    """A procedure extracted from observation."""
    name: str
    description: str
    steps: list[str]  # Parameterized steps
    parameters: list[str]  # Extracted parameters
    source_trace_id: str
    observed_from: str  # user_id of expert
    confidence: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "parameters": self.parameters,
            "source_trace_id": self.source_trace_id,
            "observed_from": self.observed_from,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


class ActionObserver:
    """
    Observes and records admin/expert actions.
    
    Captures action traces and extracts reusable procedures.
    
    Attributes:
        traces: Active traces by session_id.
        llm: LLM callable for procedure extraction.
    """
    
    # Minimum actions to consider for procedure extraction
    MIN_TRACE_LENGTH = 3
    
    # Roles that trigger observation
    OBSERVED_ROLES = {"admin", "expert", "power_user"}
    
    def __init__(
        self,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
        min_trace_length: int = 3,
    ):
        """
        Initialize action observer.
        
        Args:
            llm: LLM callable for procedure extraction.
            min_trace_length: Minimum actions for procedure.
        """
        self.llm = llm
        self.min_trace_length = min_trace_length
        self.traces: dict[str, ActionTrace] = {}
        self._extracted_procedures: list[ObservedProcedure] = []
    
    def start_session(
        self,
        session_id: str,
        user_id: str,
        user_role: str,
    ) -> Optional[str]:
        """
        Start observing a session.
        
        Args:
            session_id: Session identifier.
            user_id: User identifier.
            user_role: User's role.
            
        Returns:
            Trace ID if observation started, None otherwise.
        """
        if user_role not in self.OBSERVED_ROLES:
            return None
        
        import uuid
        trace_id = str(uuid.uuid4())
        
        self.traces[session_id] = ActionTrace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            user_role=user_role,
            actions=[],
            started_at=datetime.now(timezone.utc),
        )
        
        return trace_id
    
    def record(
        self,
        session_id: str,
        action: str,
        params: dict[str, Any],
        result: Optional[Any] = None,
    ) -> bool:
        """
        Record an action in a session.
        
        Args:
            session_id: Session identifier.
            action: Action name/type.
            params: Action parameters.
            result: Optional action result.
            
        Returns:
            True if recorded, False if session not being observed.
        """
        if session_id not in self.traces:
            return False
        
        self.traces[session_id].actions.append(Action(
            action=action,
            params=params,
            result=result,
        ))
        
        return True
    
    async def end_session(
        self,
        session_id: str,
    ) -> Optional[ObservedProcedure]:
        """
        End observation and potentially extract procedure.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Extracted procedure if applicable, None otherwise.
        """
        if session_id not in self.traces:
            return None
        
        trace = self.traces.pop(session_id)
        trace.ended_at = datetime.now(timezone.utc)
        
        # Check if trace is long enough
        if len(trace.actions) < self.min_trace_length:
            return None
        
        # Extract procedure
        procedure = await self.extract_procedure(trace)
        
        if procedure:
            self._extracted_procedures.append(procedure)
            
            audit.log_raw(
                "learning",
                "observational",
                "mml",
                "completed",
                trace_id=trace.trace_id,
                actions=len(trace.actions),
                procedure_name=procedure.name,
            )
        
        return procedure
    
    async def extract_procedure(
        self,
        trace: ActionTrace,
    ) -> Optional[ObservedProcedure]:
        """
        Extract a procedure from an action trace.
        
        Args:
            trace: Action trace to analyze.
            
        Returns:
            Extracted procedure or None.
        """
        if not self.llm:
            # Fallback: simple extraction
            return self._simple_extract(trace)
        
        # Format trace for LLM
        trace_text = self._format_trace(trace)
        
        prompt = f"""
An expert performed these steps:

{trace_text}

Convert this into a reusable procedure with:
1. A descriptive name (snake_case)
2. A brief description of what it accomplishes
3. Parameterized steps (replace specific values with parameter names like {{param_name}})
4. List of parameters needed

Format as JSON:
{{
    "name": "procedure_name",
    "description": "What this procedure does",
    "steps": ["Step 1: ...", "Step 2: ..."],
    "parameters": ["param1", "param2"]
}}
"""
        
        response = await self.llm(prompt)
        
        return self._parse_procedure(response, trace)
    
    def _format_trace(self, trace: ActionTrace) -> str:
        """Format trace for LLM prompt."""
        lines = []
        for i, action in enumerate(trace.actions, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in action.params.items())
            lines.append(f"{i}. {action.action}({params_str})")
            if action.result:
                lines.append(f"   → Result: {action.result}")
        return "\n".join(lines)
    
    def _parse_procedure(
        self,
        response: str,
        trace: ActionTrace,
    ) -> Optional[ObservedProcedure]:
        """Parse LLM response into procedure."""
        try:
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ObservedProcedure(
                    name=data.get("name", "observed_procedure"),
                    description=data.get("description", ""),
                    steps=data.get("steps", []),
                    parameters=data.get("parameters", []),
                    source_trace_id=trace.trace_id,
                    observed_from=trace.user_id,
                    confidence=0.8,  # High confidence from expert observation
                )
        except Exception as e:
            log.warning(f"Failed to parse procedure: {e}")
        
        return self._simple_extract(trace)
    
    def _simple_extract(self, trace: ActionTrace) -> ObservedProcedure:
        """Simple extraction without LLM."""
        steps = [
            f"{a.action}({', '.join(f'{k}={v}' for k, v in a.params.items())})"
            for a in trace.actions
        ]
        
        # Extract parameter names from params
        all_params = set()
        for action in trace.actions:
            all_params.update(action.params.keys())
        
        return ObservedProcedure(
            name=f"observed_{trace.actions[0].action}",
            description=f"Procedure observed from {trace.user_role}",
            steps=steps,
            parameters=list(all_params),
            source_trace_id=trace.trace_id,
            observed_from=trace.user_id,
            confidence=0.6,  # Lower confidence without LLM
        )
    
    def get_extracted_procedures(self) -> list[ObservedProcedure]:
        """Get all extracted procedures."""
        return self._extracted_procedures.copy()
    
    def get_active_sessions(self) -> list[str]:
        """Get list of active observation sessions."""
        return list(self.traces.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_action_observer: Optional[ActionObserver] = None


def get_action_observer() -> ActionObserver:
    """Get the singleton ActionObserver instance."""
    global _action_observer
    if _action_observer is None:
        _action_observer = ActionObserver()
    return _action_observer

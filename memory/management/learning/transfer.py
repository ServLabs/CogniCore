"""
Transfer Learning (Learning Type 6)

Apply knowledge from one domain to help in another.
Adapt existing procedures and patterns to new contexts.

Algorithm: Procedure adaptation + cross-domain AM edge creation
Trigger: Sleep mode (when new domain has few procedures)
LLM: Yes (expensive - domain adaptation requires reasoning)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable, Awaitable

from logger import log
from observability import audit


@dataclass
class TransferredProcedure:
    """A procedure transferred from one domain to another."""
    name: str
    source_procedure: str
    source_domain: str
    target_domain: str
    adapted_steps: list[str]
    concept_mappings: dict[str, str]  # source concept -> target concept
    notes: str  # What doesn't transfer
    confidence: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_procedure": self.source_procedure,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "adapted_steps": self.adapted_steps,
            "concept_mappings": self.concept_mappings,
            "notes": self.notes,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TransferResult:
    """Result of transfer learning."""
    procedures_transferred: int
    edges_created: int
    domains_involved: list[str]
    llm_calls: int
    duration_ms: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "procedures_transferred": self.procedures_transferred,
            "edges_created": self.edges_created,
            "domains_involved": self.domains_involved,
            "llm_calls": self.llm_calls,
            "duration_ms": self.duration_ms,
        }


class TransferLearner:
    """
    Transfers knowledge between domains.
    
    Adapts procedures and patterns from known domains
    to help bootstrap new domains.
    
    Attributes:
        llm: LLM callable for adaptation.
    """
    
    def __init__(
        self,
        llm: Optional[Callable[[str], Awaitable[str]]] = None,
        min_similarity: float = 0.5,
    ):
        """
        Initialize transfer learner.
        
        Args:
            llm: LLM callable for adaptation.
            min_similarity: Minimum domain similarity for transfer.
        """
        self.llm = llm
        self.min_similarity = min_similarity
    
    async def transfer_procedure(
        self,
        source_procedure: dict[str, Any],
        target_domain: str,
    ) -> Optional[TransferredProcedure]:
        """
        Transfer a procedure from one domain to another.
        
        Args:
            source_procedure: Source procedure dict with name, domain, steps_nl.
            target_domain: Target domain name.
            
        Returns:
            Transferred procedure or None.
        """
        if not self.llm:
            return None
        
        source_domain = source_procedure.get("domain", "unknown")
        steps = source_procedure.get("steps_nl", [])
        
        if not steps:
            return None
        
        prompt = f"""
This procedure works in the '{source_domain}' domain:

Name: {source_procedure.get('name', 'unnamed')}
Steps:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(steps))}

Adapt it for the '{target_domain}' domain:
1. Map concepts from {source_domain} to {target_domain} equivalents
2. Adjust steps for {target_domain} data structures and workflows
3. Note what doesn't transfer or needs modification

Format your response as:
ADAPTED STEPS:
1. [step]
2. [step]
...

CONCEPT MAPPINGS:
- [source concept] -> [target concept]
...

NOTES:
[What doesn't transfer or needs caution]
"""
        
        response = await self.llm(prompt)
        
        # Parse response
        adapted = self._parse_transfer_response(
            response,
            source_procedure,
            target_domain,
        )
        
        if adapted:
            audit.log_raw(
                "learning",
                "transfer",
                "mml",
                "completed",
                source=source_procedure.get("name"),
                target_domain=target_domain,
            )
        
        return adapted
    
    def _parse_transfer_response(
        self,
        response: str,
        source_procedure: dict[str, Any],
        target_domain: str,
    ) -> Optional[TransferredProcedure]:
        """Parse LLM response into TransferredProcedure."""
        lines = response.split("\n")
        
        adapted_steps = []
        concept_mappings = {}
        notes = ""
        
        section = None
        
        for line in lines:
            line = line.strip()
            
            if "ADAPTED STEPS" in line.upper():
                section = "steps"
                continue
            elif "CONCEPT MAPPING" in line.upper():
                section = "mappings"
                continue
            elif "NOTES" in line.upper():
                section = "notes"
                continue
            
            if not line:
                continue
            
            if section == "steps":
                # Extract step content
                step = line.lstrip("0123456789.-) ").strip()
                if step:
                    adapted_steps.append(step)
            elif section == "mappings":
                # Parse mapping: source -> target
                if "->" in line:
                    parts = line.split("->")
                    if len(parts) == 2:
                        source = parts[0].strip().lstrip("-• ")
                        target = parts[1].strip()
                        concept_mappings[source] = target
            elif section == "notes":
                notes += line + " "
        
        if not adapted_steps:
            return None
        
        return TransferredProcedure(
            name=f"{source_procedure.get('name', 'procedure')}_{target_domain}_adapted",
            source_procedure=source_procedure.get("name", "unknown"),
            source_domain=source_procedure.get("domain", "unknown"),
            target_domain=target_domain,
            adapted_steps=adapted_steps,
            concept_mappings=concept_mappings,
            notes=notes.strip(),
            confidence=0.7,  # Transferred procedures have lower confidence
        )
    
    async def find_transferable_procedures(
        self,
        target_domain: str,
        all_procedures: list[dict[str, Any]],
        domain_similarities: Optional[dict[str, float]] = None,
    ) -> list[dict[str, Any]]:
        """
        Find procedures that could be transferred to target domain.
        
        Args:
            target_domain: Domain needing procedures.
            all_procedures: All available procedures.
            domain_similarities: Optional pre-computed domain similarities.
            
        Returns:
            List of transferable procedures.
        """
        transferable = []
        
        for proc in all_procedures:
            source_domain = proc.get("domain", "")
            
            if source_domain == target_domain:
                continue
            
            # Check domain similarity
            if domain_similarities:
                sim = domain_similarities.get(source_domain, 0)
                if sim < self.min_similarity:
                    continue
            
            # Check if procedure is general enough
            steps = proc.get("steps_nl", [])
            if len(steps) >= 2:  # At least 2 steps
                transferable.append(proc)
        
        return transferable
    
    async def bootstrap_domain(
        self,
        target_domain: str,
        source_procedures: list[dict[str, Any]],
        max_transfers: int = 5,
    ) -> list[TransferredProcedure]:
        """
        Bootstrap a new domain with transferred procedures.
        
        Args:
            target_domain: New domain to bootstrap.
            source_procedures: Procedures from other domains.
            max_transfers: Maximum procedures to transfer.
            
        Returns:
            List of transferred procedures.
        """
        import time
        start = time.monotonic()
        
        transferred = []
        llm_calls = 0
        
        for proc in source_procedures[:max_transfers]:
            result = await self.transfer_procedure(proc, target_domain)
            llm_calls += 1
            
            if result:
                transferred.append(result)
        
        duration_ms = (time.monotonic() - start) * 1000
        
        audit.log_raw(
            "learning",
            "domain_bootstrap",
            "mml",
            "completed",
            target_domain=target_domain,
            procedures_transferred=len(transferred),
            duration_ms=duration_ms,
        )
        
        return transferred


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_transfer_learner: Optional[TransferLearner] = None


def get_transfer_learner() -> TransferLearner:
    """Get the singleton TransferLearner instance."""
    global _transfer_learner
    if _transfer_learner is None:
        _transfer_learner = TransferLearner()
    return _transfer_learner

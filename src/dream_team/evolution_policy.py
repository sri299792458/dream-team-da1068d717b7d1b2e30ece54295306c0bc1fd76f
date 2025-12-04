"""
Evolution Policy for Dream Team.

Defines the rules and constraints for agent evolution, including:
- Which agents are protected from deletion
- How to score agents for deletion (uniqueness vs overlap)
- How to assign gap concepts to existing agents
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class AgentMeta:
    """Metadata about an agent relevant for evolution decisions."""
    id: str                  # stable id / key (usually title)
    title: str
    role: str                # e.g. "lead", "coder", "domain", "data", "ml", "analysis"
    core: bool               # cannot be deleted
    tags: List[str]          # free-form: ["time_series", "biology", "feature_engineering"]
    max_focus_concepts: int = 5  # avoid overloading a single agent
    
    @property
    def tags_text(self) -> str:
        return " ".join(self.tags)


@dataclass
class EvolutionPolicyConfig:
    """Configuration for evolution policy."""
    min_per_role: Dict[str, int] = field(default_factory=dict)   # e.g. {"lead": 1, "coder": 1}
    max_team_size: int = 5
    min_team_size: int = 3
    gap_threshold: float = 0.3           
    max_concepts_per_new_agent: int = 3


class EvolutionPolicy:
    """
    Base class for evolution policies.
    """
    def __init__(self, config: EvolutionPolicyConfig):
        self.config = config
        self.agent_metas: Dict[str, AgentMeta] = {}

    def register_agent(self, agent_ref: Any, meta: AgentMeta):
        """Register metadata for an agent."""
        key = getattr(agent_ref, "title", repr(agent_ref))
        self.agent_metas[key] = meta

    def get_meta(self, agent_ref: Any) -> AgentMeta:
        """Get metadata for an agent, returning a default if not found."""
        key = getattr(agent_ref, "title", repr(agent_ref))
        if key in self.agent_metas:
            return self.agent_metas[key]
        
        # Default meta for unknown agents
        return AgentMeta(
            id=key,
            title=key,
            role="unknown",
            core=False,
            tags=[],
        )
    
    def get_all_metas(self) -> List[AgentMeta]:
        return list(self.agent_metas.values())

    def is_protected(self, meta: AgentMeta) -> bool:
        """Check if an agent is protected from deletion."""
        if meta.core:
            return True
        return False

    def can_delete(self, meta: AgentMeta, team_metas: List[AgentMeta]) -> bool:
        """Check if deleting this agent violates any team constraints."""
        if self.is_protected(meta):
            return False
            
        # Check min_per_role
        role_count = sum(1 for m in team_metas if m.role == meta.role)
        min_required = self.config.min_per_role.get(meta.role, 0)
        
        if role_count <= min_required:
            return False
            
        return True

    def score_gap_owner(
        self,
        concept: str,
        agent_meta: AgentMeta,
        depth: float,
        current_focus_count: int,
    ) -> float:
        """
        Score how well an agent fits as an owner for a gap concept.
        
        Higher score = better fit.
        """
        expertise_text = (agent_meta.title + " " + agent_meta.tags_text).lower()
        
        # simple lexical match: concept tokens in expertise
        tokens = concept.replace("_", " ").split()
        lexical_hit = any(tok in expertise_text for tok in tokens)
        
        base = depth                      # how much they already know
        bonus_lex = 0.2 if lexical_hit else 0.0
        
        # Penalty for being overloaded
        overload = max(0, current_focus_count - agent_meta.max_focus_concepts)
        penalty_load = 0.1 * overload
        
        return base + bonus_lex - penalty_load

    def compute_deletion_score(
        self,
        agent_meta: AgentMeta,
        overlap: float,
        uniqueness: float,
        load_penalty: float = 0.0
    ) -> float:
        """
        Compute a score for how "deletable" an agent is.
        Higher score = better candidate for deletion.
        
        weakness_i = a * (1 - overlap) + b * (1 - uniqueness) + c * load_penalty
        """
        # Weights
        w_overlap = 0.6
        w_unique = 0.4
        
        # If overlap is high, weakness is low.
        # If uniqueness is high, weakness is low.
        
        # Normalized inputs assumed roughly 0..1
        score = (w_overlap * (1.0 - overlap)) + (w_unique * (1.0 - uniqueness))
        
        # If agent is overloaded, maybe we want to keep them? 
        # Or maybe we want to delete them if they are bottlenecking?
        # For now, let's say load makes them LESS deletable (they are doing work).
        score -= (0.1 * load_penalty)
        
        return max(0.0, score)


class DefaultEvolutionPolicy(EvolutionPolicy):
    """Default policy implementation."""
    pass

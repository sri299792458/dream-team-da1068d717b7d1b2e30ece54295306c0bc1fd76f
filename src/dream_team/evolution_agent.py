"""
evolution_agent.py

Single, simple math-driven evolution agent for a multi-agent "dream team".

- Uses an LLM to extract 10–20 key concepts for the problem.
- Tracks per-agent expertise δ_i(v) over those concepts.
- Updates δ_i(v) with a logistic learning rule based on metric improvement.
- Computes per-agent problem overlap Ω_i(P).
- Each iteration, proposes:
    - which agent to DELETE (weakest overlap + uniqueness check via policy)
    - a SPECIALIST child based on a strong-but-generalist agent
    - a GAP expert based on under-covered concepts (routed to best owner)

The orchestrator is responsible for:
  - actually deleting old agents and creating new ones using NewAgentSpec
  - wiring math-state snippets into agent prompts if desired
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterable, Tuple
import math
from enum import Enum

from .evolution_policy import EvolutionPolicy, DefaultEvolutionPolicy, EvolutionPolicyConfig, AgentMeta

EPS = 1e-12

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def normalize_distribution(values: Dict[str, float]) -> Dict[str, float]:
    """Normalize non-negative dict to sum to 1; if sum<=0, return uniform."""
    if not values:
        return {}
    total = sum(max(0.0, v) for v in values.values())
    if total <= 0.0:
        n = len(values)
        if n == 0:
            return {}
        p = 1.0 / n
        return {k: p for k in values.keys()}
    return {k: max(0.0, v) / total for k, v in values.items()}


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient for non-negative values."""
    vals = [v for v in values if v >= 0.0]
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total <= 0.0:
        return 0.0
    vals_sorted = sorted(vals)
    cum = 0.0
    for i, x in enumerate(vals_sorted, start=1):
        cum += i * x
    g = (2.0 * cum) / (n * total) - (n + 1.0) / n
    return max(0.0, min(1.0, g))


# ---------------------------------------------------------------------------
# Problem representation
# ---------------------------------------------------------------------------

class ConceptCategory(str, Enum):
    DOMAIN = "domain"
    DATA = "data"
    MODEL = "model"
    INFRA = "infra"

@dataclass
class Concept:
    name: str
    importance: float
    category: ConceptCategory
    source: str = "llm"  # "llm" | "schema" | "code" | "fallback" | "reconstructed" | "code_dynamic"

@dataclass
class ConceptSpace:
    concepts: Dict[str, Concept] = field(default_factory=dict)

    def weights_dict(self) -> Dict[str, float]:
        """For compatibility with existing math: name -> normalized weight."""
        raw = {name: c.importance for name, c in self.concepts.items()}
        return normalize_distribution(raw)

    def concepts_list(self) -> List[str]:
        return list(self.concepts.keys())

    def get_category(self, name: str) -> Optional[ConceptCategory]:
        c = self.concepts.get(name)
        return c.category if c else None


@dataclass
class ProblemConcepts:
    """Problem as weighted concepts: concept_weights[v] = w_P(v)."""
    concept_weights: Dict[str, float] = field(default_factory=dict)

    def normalized(self) -> "ProblemConcepts":
        return ProblemConcepts(concept_weights=normalize_distribution(self.concept_weights))

    def concepts(self) -> List[str]:
        return list(self.concept_weights.keys())


# ---------------------------------------------------------------------------
# Per-agent math state
# ---------------------------------------------------------------------------

@dataclass
class AgentMathState:
    """
    Mathematical state for one agent:
      - depths[v] = δ_i(v)
      - attention[v] = θ_i(v) over problem concepts
    """
    agent_ref: Any
    depths: Dict[str, float] = field(default_factory=dict)
    attention: Dict[str, float] = field(default_factory=dict)

    # ---- summary metrics ----

    def max_depth(self) -> float:
        return max(self.depths.values()) if self.depths else 0.0

    def mean_depth(self) -> float:
        return sum(self.depths.values()) / len(self.depths) if self.depths else 0.0

    def specialization_gini(self) -> float:
        return gini_coefficient(self.depths.values())

    # ---- problem fit ----

    def problem_overlap(self, problem: ProblemConcepts) -> float:
        """
        Ω_i(P) = Σ_v δ_i(v) w_P(v) / Σ_v w_P(v)
        """
        num = 0.0
        den = 0.0
        for v, w in problem.concept_weights.items():
            d = self.depths.get(v, 0.0)
            num += d * w
            den += w
        if den <= 0.0:
            return 0.0
        return num / den

    # ---- attention & learning ----

    def update_attention(self, problem: ProblemConcepts) -> None:
        """
        Heuristic attention:
            θ_i(v) ∝ w_P(v) * (δ_i(v) + 0.05)
        """
        raw: Dict[str, float] = {}
        for v, w in problem.concept_weights.items():
            d = self.depths.get(v, 0.0)
            raw[v] = w * (d + 0.05)
        self.attention = normalize_distribution(raw)

    def update_depths_logistic(
        self,
        problem: ProblemConcepts,
        quality: float,
        alpha: float,
        beta: float,
    ) -> None:
        """
        δ update per concept v:

          learning   = α * θ(v) * quality * (1 - δ(v))
          forgetting = β * (1 - θ(v)) * δ(v)
          δ_new(v)   = δ(v) + learning - forgetting
        """
        self.update_attention(problem)
        new_depths: Dict[str, float] = dict(self.depths)

        for v in problem.concept_weights.keys():
            d = self.depths.get(v, 0.0)
            theta_v = self.attention.get(v, 0.0)

            learning = alpha * theta_v * quality * (1.0 - d)
            forgetting = beta * (1.0 - theta_v) * d

            d_new = d + learning - forgetting
            new_depths[v] = d_new

        self.depths = new_depths


# ---------------------------------------------------------------------------
# Team-level math state
# ---------------------------------------------------------------------------

@dataclass
class TeamMathState:
    """
    Team-level math state:
      - shared ProblemConcepts
      - per-agent AgentMathState
      - metric history, learning hyperparameters
    """
    problem: ProblemConcepts
    agent_states: Dict[str, AgentMathState] = field(default_factory=dict)
    metric_history: List[float] = field(default_factory=list)
    minimize_metric: bool = True
    alpha: float = 0.5
    beta: float = 0.1

    # ---- metric & quality ----

    def record_metric(self, value: float) -> None:
        self.metric_history.append(value)

    def _compute_quality(self) -> float:
        """
        Map recent metric change to quality scalar Q:
          improved → 0.8
          plateau  → 0.5
          worse    → 0.3
        """
        if len(self.metric_history) < 2:
            return 0.5
        prev, cur = self.metric_history[-2], self.metric_history[-1]
        if self.minimize_metric:
            improvement = prev - cur
        else:
            improvement = cur - prev
        if improvement > 0:
            return 0.8
        if abs(improvement) < 1e-3:
            return 0.5
        return 0.3

    # ---- dynamics ----

    def update_all_depths(self) -> float:
        """Update δ for all agents and return quality used."""
        quality = self._compute_quality()
        for s in self.agent_states.values():
            s.update_depths_logistic(self.problem, quality, self.alpha, self.beta)
        return quality

    # ---- team coverage & selection ----

    def team_coverage(self) -> Dict[str, float]:
        """cov(v) = max_i δ_i(v)."""
        coverage = {c: 0.0 for c in self.problem.concepts()}
        for s in self.agent_states.values():
            for c in coverage.keys():
                coverage[c] = max(coverage[c], s.depths.get(c, 0.0))
        return coverage

    def compute_uniqueness(self) -> Dict[str, float]:
        """
        Compute uniqueness contribution for each agent.
        unique_i = Σ_v w_P(v) * max(δ_i(v) - second_best(v), 0)
        """
        uniqueness = {}
        
        # Precompute second best depth for each concept
        second_best_depths = {}
        for v in self.problem.concepts():
            all_depths = sorted([s.depths.get(v, 0.0) for s in self.agent_states.values()], reverse=True)
            if len(all_depths) > 1:
                second_best_depths[v] = all_depths[1]
            else:
                second_best_depths[v] = 0.0
        
        for key, state in self.agent_states.items():
            u = 0.0
            for v, w in self.problem.concept_weights.items():
                d = state.depths.get(v, 0.0)
                sb = second_best_depths.get(v, 0.0)
                u += w * max(0.0, d - sb)
            uniqueness[key] = u
            
        # Normalize uniqueness to 0-1 range for easier scoring
        max_u = max(uniqueness.values()) if uniqueness else 1.0
        if max_u > 0:
            for k in uniqueness:
                uniqueness[k] /= max_u
                
        return uniqueness

    def select_weakest_agent(self, policy: EvolutionPolicy) -> Optional[AgentMathState]:
        """
        Select weakest agent based on policy (overlap + uniqueness + constraints).
        """
        if not self.agent_states:
            return None
            
        uniqueness = self.compute_uniqueness()
        team_metas = policy.get_all_metas()
        
        candidates = []
        for key, state in self.agent_states.items():
            meta = policy.get_meta(state.agent_ref)
            
            # Check constraints
            if not policy.can_delete(meta, team_metas):
                continue
                
            overlap = state.problem_overlap(self.problem)
            uniq = uniqueness.get(key, 0.0)
            
            # Simple load proxy: count concepts with depth > 0.4
            load = sum(1 for d in state.depths.values() if d > 0.4)
            
            weakness = policy.compute_deletion_score(
                agent_meta=meta,
                overlap=overlap,
                uniqueness=uniq,
                load_penalty=load
            )
            candidates.append((weakness, state))
            
        if not candidates:
            return None
            
        # Highest weakness score is the best candidate for deletion
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def select_strong_generalist(
        self,
        overlap_threshold: float,
        gini_threshold: float,
    ) -> Optional[AgentMathState]:
        """
        Agent with high overlap but low specialization (good candidate to specialize).
        """
        best: Optional[AgentMathState] = None
        best_overlap = -1.0
        for s in self.agent_states.values():
            overlap = s.problem_overlap(self.problem)
            gini = s.specialization_gini()
            if overlap >= overlap_threshold and gini <= gini_threshold:
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = s
        return best


# ---------------------------------------------------------------------------
# Evolution decisions
# ---------------------------------------------------------------------------

@dataclass
class NewAgentSpec:
    """
    Specification for creating a new agent.

    The orchestrator can turn this into an actual Agent(...) instance.
    """
    kind: str                  # "specialize" or "gap"
    title: str
    expertise: str
    role: str
    focus_concepts: List[str]


@dataclass
class EvolutionDecision:
    """
    Output of one evolution step.

    - quality: scalar Q used for depth updates (for logging)
    - agents_to_delete: underlying Agent objects that should be removed
    - new_agent_specs: specs describing new agents to create
    - debug_info: useful diagnostics
    """
    quality: float
    agents_to_delete: List[Any]
    new_agent_specs: List[NewAgentSpec]
    debug_info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Single Evolution Agent
# ---------------------------------------------------------------------------

class EvolutionAgent:
    """
    EvolutionAgent:
      - lives from the beginning
      - holds TeamMathState
      - every iteration:
          * updates δ via logistic rule
          * chooses deletion candidate
          * proposes specialist + gap expert specs
    """

    def __init__(
        self,
        llm: Any,
        target_team_size: Tuple[int, int] = (3, 6),
        gap_threshold: float = 0.3,
        specialize_overlap_threshold: float = 0.4,
        specialize_gini_threshold: float = 0.5,
        alpha: float = 0.5,
        beta: float = 0.1,
        policy: Optional[EvolutionPolicy] = None,
    ):
        self.llm = llm
        self.target_team_size = target_team_size
        self.gap_threshold = gap_threshold
        self.specialize_overlap_threshold = specialize_overlap_threshold
        self.specialize_gini_threshold = specialize_gini_threshold
        self.alpha = alpha
        self.beta = beta
        
        # Initialize policy with default config if not provided
        if policy is None:
            config = EvolutionPolicyConfig(
                min_per_role={},
                max_team_size=target_team_size[1],
                min_team_size=target_team_size[0],
                gap_threshold=gap_threshold
            )
            self.policy = DefaultEvolutionPolicy(config)
        else:
            self.policy = policy
            
            self.policy = policy
            
        self.team_state: Optional[TeamMathState] = None
        self._concept_space: Optional[ConceptSpace] = None

    # ----- LLM concept extraction -----

    def define_problem_space(
        self,
        problem_statement: str,
        target_metric: str,
        max_concepts: int = 7,
        exploration_context: Optional[str] = None,
    ) -> ProblemConcepts:
        """
        Define the problem space P by extracting key domain concepts.
        
        Simplified to focus only on strategic domain concepts that drive
        agent specialization, not mechanical schema or technique extraction.
        
        Args:
            problem_statement: The research problem
            target_metric: Metric being optimized
            max_concepts: Maximum concepts to extract (default: 7)
            exploration_context: Optional context from bootstrap exploration
        
        Returns:
            ProblemConcepts with normalized weights
        """
        domain = self._build_domain_concepts_from_llm(
            problem_statement, 
            target_metric, 
            max_concepts,
            exploration_context
        )

        if not domain:
            # fallback: use target_metric name
            domain[target_metric.lower()] = Concept(
                name=target_metric.lower(),
                importance=3.0,
                category=ConceptCategory.DOMAIN,
                source="fallback",
            )

        # Normalize importance to get weights
        weights = normalize_distribution({n: c.importance for n, c in domain.items()})

        for name, w in weights.items():
            domain[name].importance = w

        # Stash full ConceptSpace
        self._concept_space = ConceptSpace(concepts=domain)

        # Build ProblemConcepts for compatibility
        return ProblemConcepts(
            concept_weights={name: c.importance for name, c in domain.items()}
        )

    def _build_domain_concepts_from_llm(
        self, 
        problem_statement: str, 
        target_metric: str, 
        max_concepts: int,
        exploration_context: Optional[str] = None
    ) -> Dict[str, Concept]:
        """Extract domain concepts via LLM."""
        context_section = ""
        if exploration_context:
            context_section = f"\n## Bootstrap Context:\n{exploration_context[:500]}\n"
        
        prompt = f"""Identify 5-7 core technical concepts for this research problem.

Problem:
{problem_statement}

Target Metric: {target_metric}
{context_section}

Identify the MOST CRITICAL technical concepts. Focus on distinct areas of expertise.

Output valid JSON only:
[
  {{"concept": "ensemble_methods", "importance": 2}},
  {{"concept": "feature_engineering", "importance": 3}}
]

Rules:
- importance: 1 (nice-to-have), 2 (important), 3 (critical)
- 5-7 concepts total
- No markdown, no other text
"""

        try:
            raw_concepts = self.llm.generate_json(prompt, temperature=0.2)
            if not isinstance(raw_concepts, list):
                raw_concepts = []
        except Exception:
            raw_concepts = []

        domain_concepts: Dict[str, Concept] = {}

        for item in raw_concepts:
            if not isinstance(item, dict):
                continue
            
            name = item.get("concept")
            if not name:
                continue
                
            importance = float(item.get("importance", 2.0))
            
            domain_concepts[name] = Concept(
                name=name,
                importance=importance,
                category=ConceptCategory.DOMAIN,
                source="llm",
            )
            
            if len(domain_concepts) >= max_concepts:
                break
                
        return domain_concepts

    # Schema and technique builders removed - concepts now come only from domain analysis

    # ----- Initialization & Updates -----

    def initialize(
        self,
        agents: List[Any],
        problem_statement: str,
        target_metric: str,
        minimize_metric: bool = True,
        column_schemas: Optional[Dict[str, List[str]]] = None,
        techniques: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize team math state from agents + problem.
        Simple init:
          - δ_i(v) = 0.3 if concept tokens appear in agent.expertise/title, else 0.05.
        """
        problem = self.define_problem_space(
            problem_statement=problem_statement,
            target_metric=target_metric,
            column_schemas=column_schemas,
            techniques=techniques
        )
        problem = problem.normalized()

        agent_states: Dict[str, AgentMathState] = {}
        for agent in agents:
            key = getattr(agent, "title", repr(agent))
            state = AgentMathState(agent_ref=agent)

            expertise_text = (
                (getattr(agent, "expertise", "") or "") + " " +
                (getattr(agent, "title", "") or "")
            ).lower().replace("-", " ").replace("_", " ")

            for concept in problem.concept_weights.keys():
                tokens = concept.replace("_", " ").split()
                if any(tok in expertise_text for tok in tokens):
                    d0 = 0.8
                else:
                    d0 = 0.05
                state.depths[concept] = d0

            agent_states[key] = state

        self.team_state = TeamMathState(
            problem=problem,
            agent_states=agent_states,
            metric_history=[],
            minimize_metric=minimize_metric,
            alpha=self.alpha,
            beta=self.beta,
        )

    def update_team(self, agents: List[Any]) -> None:
        """
        Update team composition while preserving existing agent states and metric history.
        """
        if self.team_state is None:
            raise RuntimeError("EvolutionAgent not initialized. Call initialize() first.")

        current_states = self.team_state.agent_states
        new_states: Dict[str, AgentMathState] = {}
        
        problem = self.team_state.problem

        for agent in agents:
            key = getattr(agent, "title", repr(agent))
            
            if key in current_states:
                # Preserve existing state, but update ref in case object changed
                state = current_states[key]
                state.agent_ref = agent
                new_states[key] = state
            else:
                # Initialize new agent
                state = AgentMathState(agent_ref=agent)
                expertise_text = (
                    (getattr(agent, "expertise", "") or "") + " " +
                    (getattr(agent, "title", "") or "")
                ).lower().replace("-", " ").replace("_", " ")

                for concept in problem.concept_weights.keys():
                    tokens = concept.replace("_", " ").split()
                    if any(tok in expertise_text for tok in tokens):
                        d0 = 0.8
                    else:
                        d0 = 0.05
                    state.depths[concept] = d0
                
                new_states[key] = state

        self.team_state.agent_states = new_states


    # ----- Per-iteration API -----

    def record_metric(self, value: float) -> None:
        """Call once per iteration with the current scalar metric value."""
        if self.team_state is None:
            raise RuntimeError("EvolutionAgent not initialized")
        self.team_state.record_metric(value)

    def _redistribute_knowledge_before_deletion(self, deleted_agent: Any) -> None:
        """
        Transfer deleted agent's knowledge to most related remaining agents.
        
        Prevents knowledge loss when pruning team - redistributes techniques,
        patterns, and papers to agents with overlapping expertise.
        """
        if not self.team_state:
            return
        
        # Get deleted agent's KB
        if not hasattr(deleted_agent, 'knowledge_base'):
            return
        
        deleted_kb = deleted_agent.knowledge_base
        deleted_title = getattr(deleted_agent, 'title', 'Agent')
        
        # Find most related agents based on concept overlap
        deleted_state = None
        for state in self.team_state.agent_states.values():
            if state.agent_ref == deleted_agent:
                deleted_state = state
                break
        
        if not deleted_state:
            return
        
        # Score remaining agents by concept overlap with deleted agent
        overlap_scores = []
        for key, state in self.team_state.agent_states.items():
            if state.agent_ref == deleted_agent:
                continue  # Skip self
            
            # Compute depth overlap
            overlap = sum(
                min(deleted_state.depths.get(c, 0), state.depths.get(c, 0))
                for c in deleted_state.depths.keys()
            )
            overlap_scores.append((overlap, state.agent_ref))
        
        if not overlap_scores:
            return
        
        # Sort by overlap, take top 2
        overlap_scores.sort(reverse=True, key=lambda x: x[0])
        recipients = [agent for _, agent in overlap_scores[:2]]
        
        # Transfer knowledge
        print(f"      📚 Redistributing {deleted_title}'s knowledge to {len(recipients)} agents...")
        
        for recipient in recipients:
            if not hasattr(recipient, 'knowledge_base'):
                continue
            
            # Transfer techniques
            for technique in deleted_kb.techniques_mastered:
                if technique not in recipient.knowledge_base.techniques_mastered:
                    recipient.knowledge_base.techniques_mastered.append(technique)
            
            # Transfer successful patterns
            for pattern in deleted_kb.successful_patterns:
                if pattern not in recipient.knowledge_base.successful_patterns:
                    recipient.knowledge_base.successful_patterns.append(pattern)
            
            # Transfer papers
            existing_titles = [p.title for p in recipient.knowledge_base.papers]
            for paper in deleted_kb.papers:
                if paper.title not in existing_titles:
                    recipient.knowledge_base.add_paper(paper)
            
            print(f"         → Transferred to {getattr(recipient, 'title', 'Agent')}")

    def step(self) -> EvolutionDecision:
        """
        Perform one evolution step.

        - if not enough history: no change
        - else:
            * update δ for all agents
            * compute coverage and gaps
            * propose:
                - deletion candidate (weakest overlap)
                - specialist (if any strong generalist)
                - gap expert (if any gaps)
        """
        if self.team_state is None:
            raise RuntimeError("EvolutionAgent not initialized")

        # need at least 2 data points to compute improvement
        if len(self.team_state.metric_history) < 2:
            return EvolutionDecision(
                quality=0.5,
                agents_to_delete=[],
                new_agent_specs=[],
                debug_info={"note": "not enough metric history yet"},
            )

        # 1) update δ using logistic rule
        quality = self.team_state.update_all_depths()
        problem = self.team_state.problem

        # 2) coverage & gaps
        coverage = self.team_state.team_coverage()
        gaps = [c for c, cov in coverage.items() if cov < self.gap_threshold]

        # Check current team size for constraints
        current_size = len(self.team_state.agent_states)
        min_size, max_size = self.target_team_size

        # 3) deletion candidate: weakest overlap (only if we're above min size)
        agents_to_delete: List[Any] = []
        if current_size > min_size:
            weakest_state = self.team_state.select_weakest_agent(self.policy)
            if weakest_state is not None:
                # Before deleting, redistribute their knowledge to related agents
                self._redistribute_knowledge_before_deletion(weakest_state.agent_ref)
                agents_to_delete.append(weakest_state.agent_ref)

        # 4) New agent proposals (only if we're below max size OR if we're deleting someone)
        new_specs: List[NewAgentSpec] = []
        can_add = (current_size < max_size) or (len(agents_to_delete) > 0)

        if can_add:
            # Track projected size to enforce max limit
            projected_size = current_size - len(agents_to_delete)

            # 4a) specialization candidate: strong generalist
            if projected_size < max_size:
                best_specialize = self.team_state.select_strong_generalist(
                    overlap_threshold=self.specialize_overlap_threshold,
                    gini_threshold=self.specialize_gini_threshold,
                )

                if best_specialize is not None:
                    s = best_specialize
                    sorted_concepts = sorted(
                        s.depths.items(), key=lambda x: x[1], reverse=True
                    )
                    top_k = [c for c, d in sorted_concepts[:3]]
                    parent_title = getattr(s.agent_ref, "title", "Agent")
                    title = f"{parent_title} Specialist"
                    expertise = f"Specialist focusing on: {', '.join(top_k)}"
                    role = "Provide deep, focused expertise on these key concepts for the current problem."
                    new_specs.append(
                        NewAgentSpec(
                            kind="specialize",
                            title=title,
                            expertise=expertise,
                            role=role,
                            focus_concepts=top_k,
                        )
                    )
                    projected_size += 1

            # 4b) gap-based new agent (using policy routing)
            if gaps and projected_size < max_size:
                assignments, orphans = self._assign_gaps_to_owners(gaps)
                
                # For each owner, create specialists or boost existing
                for owner_key, concepts in assignments.items():
                    if projected_size >= max_size:
                        break

                    owner_state = self.team_state.agent_states[owner_key]
                    owner_meta = self.policy.get_meta(owner_state.agent_ref)
                    
                    groups = self._group_concepts(concepts, self.policy.config.max_concepts_per_new_agent)
                    
                    # Prevent redundant specialization if owner is already a specialist
                    if "Specialist" in owner_meta.title:
                        continue
                    
                    for group in groups:
                        if projected_size >= max_size:
                            break

                        # Avoid recursive naming if parent is already a Gap Specialist
                        if "Gap Specialist" in owner_meta.title:
                            title = f"Gap Specialist ({', '.join(group)})"
                        else:
                            title = f"{owner_meta.title} ({', '.join(group)}) Specialist"

                        expertise = f"Specialist focusing on: {', '.join(group)}"
                        role = "Provide deep, focused expertise on these concepts."
                        new_specs.append(NewAgentSpec(
                            kind="specialize_gap",
                            title=title,
                            expertise=expertise,
                            role=role,
                            focus_concepts=group
                        ))
                        projected_size += 1
                
                # For orphan concepts, create small gap agents
                if orphans and projected_size < max_size:
                    # Sort by importance
                    sorted_orphans = sorted(
                        orphans, 
                        key=lambda c: problem.concept_weights.get(c, 0.0), 
                        reverse=True
                    )
                    groups = self._group_concepts(sorted_orphans, self.policy.config.max_concepts_per_new_agent)
                    
                    for group in groups:
                        if projected_size >= max_size:
                            break

                        title = f"Gap Specialist ({', '.join(group)})"
                        expertise = "Domain expert created to cover under-served concepts: " + ", ".join(group)
                        role = "Introduce methods and knowledge centered on these concepts."
                        new_specs.append(NewAgentSpec(
                            kind="gap",
                            title=title,
                            expertise=expertise,
                            role=role,
                            focus_concepts=group
                        ))
                        projected_size += 1

        debug = {
            "quality": quality,
            "coverage": coverage,
            "gaps": gaps,
            "team_size": current_size,
            "min_size": min_size,
            "max_size": max_size,
        }

        return EvolutionDecision(
            quality=quality,
            agents_to_delete=agents_to_delete,
            new_agent_specs=new_specs,
            debug_info=debug,
        )
    
    def _assign_gaps_to_owners(self, gaps: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Assign gap concepts to existing agents who are best suited to own them.
        Returns:
            assignments: {agent_key: [concepts]}
            orphans: [concepts] (no suitable owner found)
        """
        assignments: Dict[str, List[str]] = {}
        orphans: List[str] = []
        
        if not self.team_state:
            return {}, gaps
            
        for concept in gaps:
            best_agent_key = None
            best_score = -1.0
            
            for key, state in self.team_state.agent_states.items():
                meta = self.policy.get_meta(state.agent_ref)
                
                # Count current high-focus concepts
                current_focus = sum(1 for d in state.depths.values() if d > 0.4)
                
                score = self.policy.score_gap_owner(
                    concept=concept,
                    agent_meta=meta,
                    depth=state.depths.get(concept, 0.0),
                    current_focus_count=current_focus
                )
                
                if score > best_score:
                    best_score = score
                    best_agent_key = key
            
            # Threshold for ownership (e.g. 0.2 means at least some relevance)
            if best_agent_key and best_score > 0.2:
                if best_agent_key not in assignments:
                    assignments[best_agent_key] = []
                assignments[best_agent_key].append(concept)
            else:
                orphans.append(concept)
                
        return assignments, orphans

    def _group_concepts(self, concepts: List[str], max_per_group: int) -> List[List[str]]:
        """Group concepts into chunks of max size."""
        concepts = sorted(concepts)  # deterministic
        groups = []
        for i in range(0, len(concepts), max_per_group):
            groups.append(concepts[i:i+max_per_group])
        return groups
    
    # ----- Integration with 5-layer architecture -----
    
    def refine_concepts_from_code(self, agent: Any, techniques: List[str], boost: float = 0.1) -> None:
        """
        Refine agent concept depths based on techniques used in code.
        
        If a technique matches a concept, bump δ for that agent-concept pair.
        This provides feedback from actual code execution to the mathematical model.
        
        Args:
            agent: Agent whose depths to update
            techniques: List of techniques from CodeAnalysis
            boost: How much to boost matching concept depths (default: 0.1)
        """
        if self.team_state is None:
            return
        
        agent_key = getattr(agent, "title", repr(agent))
        if agent_key not in self.team_state.agent_states:
            return
        
        agent_state = self.team_state.agent_states[agent_key]
        
        # Match techniques to concepts
        for technique in techniques:
            technique_lower = technique.lower().replace(" ", "_")
            
            for concept in agent_state.depths.keys():
                # Check if technique matches concept
                if technique_lower in concept or concept in technique_lower:
                    # Boost this concept depth
                    current_depth = agent_state.depths[concept]
                    agent_state.depths[concept] = min(1.0, current_depth + boost)
    
    def get_coverage_and_gaps(self) -> Tuple[Dict[str, float], List[str]]:
        """
        Get team coverage and gaps for ContextBuilder integration.
        
        Returns:
            Tuple of (coverage_dict, gaps_list)
        """
        if self.team_state is None:
            return {}, []
        
        coverage = self.team_state.team_coverage()
        gaps = [c for c, cov in coverage.items() if cov < self.gap_threshold]
        
        return coverage, gaps
    
    def seed_agent_knowledge(
        self,
        agent: Any,
        iteration_records: List[Any],  # List[IterationRecord]
        focus_concepts: List[str]
    ) -> None:
        """
        Seed a newly created agent's knowledge base with relevant past learnings.
        
        Extracts techniques and patterns from past iterations that match the
        agent's focus concepts.
        
        Args:
            agent: Newly created agent
            iteration_records: Past IterationRecords from the experiment
            focus_concepts: Concepts this agent should focus on
        """
        if not iteration_records:
            return
        
        kb = agent.knowledge_base
        
        # Extract relevant techniques from past iterations
        relevant_techniques = set()
        relevant_patterns = []
        
        for iter_rec in iteration_records:
            # Check if any code techniques match focus concepts
            for technique in iter_rec.code_analysis.techniques:
                technique_lower = technique.lower().replace(" ", "_")
                
                for concept in focus_concepts:
                    if technique_lower in concept or concept in technique_lower:
                        relevant_techniques.add(technique)
                        
                        # If this iteration was successful, add as pattern
                        if iter_rec.metrics and len(iter_rec.metrics) > 0:
                            # Get any metric value as proxy for success
                            metric_val = list(iter_rec.metrics.values())[0]
                            if metric_val is not None:
                                relevant_patterns.append(
                                    f"Iter {iter_rec.iteration}: {technique} achieved {list(iter_rec.metrics.keys())[0]}={metric_val:.4f}"
                                )
        
        # Populate agent KB
        for tech in relevant_techniques:
            kb.add_technique(tech)
        
        for pattern in relevant_patterns[-5:]:  # Last 5 relevant patterns
            kb.successful_patterns.append(pattern)

    def refine_concept_space(
        self,
        problem_statement: str,
        target_metric: str,
        last_approach: str,
        last_metric: float,
        improved: bool,
        reflection_text: str,
        iteration: int
    ) -> None:
        """
        Refine the concept space based on latest iteration learnings.
        
        Called after each iteration to dynamically evolve which concepts
        matter based on what the team actually tried and what the PI reflected on.
        """
        if not self._concept_space or not self.team_state or iteration == 0:
            return
        
        # Format current concepts with coverage
        current_concepts_str = "\n".join([
            f"{name}: importance={c.importance:.2f}, coverage={self.team_state.concept_coverage().get(name, 0):.2f}"
            for name, c in self._concept_space.concepts.items()
        ])
        
        # Get reflection snippet
        reflection_snippet = reflection_text[:300] + "..." if len(reflection_text) > 300 else reflection_text
        
        prompt = f"""
Identify 5-7 core technical concepts for this research problem.

## Problem:
{problem_statement}

## Latest Iteration (#{iteration}):
Tried: {last_approach[:200]}
Result: {target_metric} = {last_metric:.4f} ({'improved' if improved else 'no improvement'})
PI noted: {reflection_snippet}

## Current Concepts:
{current_concepts_str}

## Task:
What expertise areas matter MOST for this specific problem?

Output (one per line):
concept_name | importance (1-3)

Keep concepts that remain relevant.
Add concepts that emerged as important.
Remove concepts (importance=0) that proved irrelevant.
""".strip()
        
        raw = self.llm.generate(prompt, temperature=0.2)
        new_concepts: Dict[str, Concept] = {}
        
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            
            name = parts[0]
            if not name:
                continue
            
            importance = 2.0
            if len(parts) >= 2:
                try:
                    importance = float(parts[1])
                except ValueError:
                    importance = 2.0
            
            # Skip if marked for removal
            if importance <= 0:
                continue
            
            new_concepts[name] = Concept(
                name=name,
                importance=importance,
                category=ConceptCategory.DOMAIN,
                source="refinement",
            )
            
            if len(new_concepts) >= 7:
                break
        
        if not new_concepts:
            # Keep existing if refinement failed
            return
        
        # Replace concept space
        self._concept_space = ConceptSpace(concepts=new_concepts)
        
        # Normalize
        weights = normalize_distribution({n: c.importance for n, c in new_concepts.items()})
        for n, w in weights.items():
            self._concept_space.concepts[n].importance = w
        
        # Update team state problem weights
        self.team_state.problem = ProblemConcepts(
            concept_weights={n: c.importance for n, c in new_concepts.items()}
        )
        
        # Update agent depths for new/removed concepts
        for state in self.team_state.agent_states.values():
            # Remove depths for concepts no longer tracked
            removed = [n for n in state.depths if n not in new_concepts]
            for n in removed:
                del state.depths[n]  
            
            # Initialize depths for new concepts
            expertise_text = (
                (getattr(state.agent_ref, "expertise", "") or "") + " " +
                (getattr(state.agent_ref, "title", "") or "")
            ).lower().replace("-", " ").replace("_", " ")
            
            for name in new_concepts.keys():
                if name not in state.depths:
                    tokens = name.replace("_", " ").split()
                    if any(tok in expertise_text for tok in tokens):
                        state.depths[name] = 0.3
                    else:
                        state.depths[name] = 0.05

    def maybe_expand_concepts_from_techniques(
        self,
        techniques: List[str],
        min_occurrences: int = 2
    ) -> None:
        """
        Expand the concept space with new model/technique concepts
        based on frequently used techniques.
        """
        if self.team_state is None:
            return

        if not hasattr(self, "_concept_space") or self._concept_space is None:
            # Reconstruct from current problem weights if needed
            self._concept_space = ConceptSpace(
                concepts={
                    name: Concept(
                        name=name,
                        importance=w,
                        category=ConceptCategory.MODEL,  # fallback
                        source="reconstructed",
                    )
                    for name, w in self.team_state.problem.concept_weights.items()
                }
            )

        # Count occurrences
        from collections import Counter
        counter = Counter(t.lower().replace(" ", "_") for t in techniques)

        new_concepts = {}
        for name, count in counter.items():
            if count < min_occurrences:
                continue
            if name in self._concept_space.concepts:
                continue

            new_concepts[name] = Concept(
                name=name,
                importance=0.5 * max(
                    (c.importance for c in self._concept_space.concepts.values()), default=1.0
                ),
                category=ConceptCategory.MODEL,
                source="code_dynamic",
            )

        if not new_concepts:
            return

        # Merge into concept space
        self._concept_space.concepts.update(new_concepts)

        # Re-normalize weights
        weights = normalize_distribution({
            n: c.importance for n, c in self._concept_space.concepts.items()
        })
        for n, w in weights.items():
            self._concept_space.concepts[n].importance = w

        # Update problem weights used by TeamMathState
        self.team_state.problem = ProblemConcepts(
            concept_weights={n: c.importance for n, c in self._concept_space.concepts.items()}
        )

        # Initialize depths for the new concepts for all agents
        for state in self.team_state.agent_states.values():
            expertise_text = (
                (getattr(state.agent_ref, "expertise", "") or "") + " " +
                (getattr(state.agent_ref, "title", "") or "")
            ).lower().replace("-", " ").replace("_", " ")

            for name in new_concepts.keys():
                if name in state.depths:
                    continue
                tokens = name.replace("_", " ").split()
                if any(tok in expertise_text for tok in tokens):
                    d0 = 0.3
                else:
                    d0 = 0.05
                state.depths[name] = d0

    # ----- Serialization -----

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full state to dictionary"""
        state = {
            "target_team_size": self.target_team_size,
            "gap_threshold": self.gap_threshold,
            "specialize_overlap_threshold": self.specialize_overlap_threshold,
            "specialize_gini_threshold": self.specialize_gini_threshold,
            "alpha": self.alpha,
            "beta": self.beta,
        }
        
        if self.team_state:
            # Serialize TeamMathState
            ts = self.team_state
            
            # Serialize agent states
            agent_math_states = {}
            for k, v in ts.agent_states.items():
                agent_math_states[k] = {
                    "depths": v.depths,
                    "attention": v.attention
                }
                
            state["team_state"] = {
                "problem_weights": ts.problem.concept_weights,
                "metric_history": ts.metric_history,
                "minimize_metric": ts.minimize_metric,
                "alpha": ts.alpha,
                "beta": ts.beta,
                "agent_math_states": agent_math_states
            }
            
        return state

    @classmethod
    def from_dict(cls, data: Dict[str, Any], llm: Any) -> 'EvolutionAgent':
        """Restore from dictionary"""
        agent = cls(
            llm=llm,
            target_team_size=tuple(data.get("target_team_size", [3, 6])),
            gap_threshold=data.get("gap_threshold", 0.2),
            specialize_overlap_threshold=data.get("specialize_overlap_threshold", 0.6),
            specialize_gini_threshold=data.get("specialize_gini_threshold", 0.3),
            alpha=data.get("alpha", 0.5),
            beta=data.get("beta", 0.1)
        )
        
        if "team_state" in data:
            ts_data = data["team_state"]
            
            # Reconstruct ProblemConcepts
            problem = ProblemConcepts(concept_weights=ts_data.get("problem_weights", {}))
            
            # Reconstruct AgentMathStates (without refs initially)
            agent_states = {}
            for k, v in ts_data.get("agent_math_states", {}).items():
                # We create a dummy state, refs will be linked in update_team
                s = AgentMathState(agent_ref=None)
                s.depths = v.get("depths", {})
                s.attention = v.get("attention", {})
                agent_states[k] = s
                
            agent.team_state = TeamMathState(
                problem=problem,
                agent_states=agent_states,
                metric_history=ts_data.get("metric_history", []),
                minimize_metric=ts_data.get("minimize_metric", True),
                alpha=ts_data.get("alpha", 0.5),
                beta=ts_data.get("beta", 0.1)
            )
            
        return agent

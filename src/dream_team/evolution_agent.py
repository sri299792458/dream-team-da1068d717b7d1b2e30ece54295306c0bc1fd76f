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
        max_concepts: int = 20,
        column_schemas: Optional[Dict[str, List[str]]] = None,
        techniques: Optional[List[str]] = None,
    ) -> ProblemConcepts:
        """
        Define the problem space P by extracting key concepts and their importance
        from multiple sources: LLM (domain), Schema (data), and Code (model).
        """
        domain = self._build_domain_concepts_from_llm(problem_statement, target_metric, max_concepts)
        data = self._build_data_concepts_from_schema(column_schemas)
        model = self._build_model_concepts_from_techniques(techniques)

        all_concepts: Dict[str, Concept] = {}
        for d in (domain, data, model):
            for name, c in d.items():
                if name in all_concepts:
                    # keep max importance, prefer more specific category if needed
                    existing = all_concepts[name]
                    existing.importance = max(existing.importance, c.importance)
                    # Could merge categories/sources here if needed
                else:
                    all_concepts[name] = c

        if not all_concepts:
            # fallback: use target_metric name
            all_concepts[target_metric.lower()] = Concept(
                name=target_metric.lower(),
                importance=3.0,
                category=ConceptCategory.MODEL,
                source="fallback",
            )

        # Normalize importance to get weights
        weights = normalize_distribution({n: c.importance for n, c in all_concepts.items()})

        for name, w in weights.items():
            all_concepts[name].importance = w

        # Stash full ConceptSpace
        self._concept_space = ConceptSpace(concepts=all_concepts)

        # Build ProblemConcepts for compatibility
        return ProblemConcepts(
            concept_weights={name: c.importance for name, c in all_concepts.items()}
        )

    def _build_domain_concepts_from_llm(
        self, problem_statement: str, target_metric: str, max_concepts: int
    ) -> Dict[str, Concept]:
        prompt = f"""
You are helping configure a multi-agent research team for this problem.

Problem:
{problem_statement}

Target metric: {target_metric}

Task:
Extract the 10-20 most important technical concepts, domain ideas, or methods
that the team should explicitly track.

Rules:
- Output ONE concept per line.
- Format: concept_name | importance | category
- concept_name: short, snake_case, no spaces (e.g. gradient_boosting, microbial_growth)
- importance: integer 1, 2, or 3 (3 = very important).
- category: one of {{domain, data, model, infra}}. If unsure, use domain.
- No explanations, no headings, only the raw list.
""".strip()

        raw = self.llm.generate(prompt, temperature=0.2)
        domain_concepts: Dict[str, Concept] = {}

        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            
            name = parts[0]
            if not name:
                continue
                
            importance = 1.0
            category = ConceptCategory.DOMAIN
            
            if len(parts) >= 2:
                try:
                    importance = float(parts[1])
                except ValueError:
                    importance = 1.0
            
            if len(parts) >= 3:
                cat_str = parts[2].lower()
                # Check if valid category
                if cat_str in [e.value for e in ConceptCategory]:
                    category = ConceptCategory(cat_str)
            
            domain_concepts[name] = Concept(
                name=name,
                importance=importance,
                category=category,
                source="llm",
            )
            
            if len(domain_concepts) >= max_concepts:
                break
                
        return domain_concepts

    def _build_data_concepts_from_schema(
        self, column_schemas: Optional[Dict[str, List[str]]]
    ) -> Dict[str, Concept]:
        data_concepts = {}
        if not column_schemas:
            return data_concepts

        has_datetime = False
        has_categorical = False

        for df_name, cols in column_schemas.items():
            for col in cols:
                col_lower = col.lower()
                if any(tok in col_lower for tok in ["date", "time", "timestamp"]):
                    has_datetime = True
                if any(tok in col_lower for tok in ["category", "type", "code", "class", "status"]):
                    has_categorical = True

        if has_datetime:
            data_concepts["time_series_features"] = Concept(
                name="time_series_features",
                importance=2.0,
                category=ConceptCategory.DATA,
                source="schema",
            )
        if has_categorical:
            data_concepts["categorical_encoding"] = Concept(
                name="categorical_encoding",
                importance=2.0,
                category=ConceptCategory.DATA,
                source="schema",
            )
        return data_concepts

    def _build_model_concepts_from_techniques(
        self, techniques: Optional[List[str]]
    ) -> Dict[str, Concept]:
        model_concepts = {}
        if not techniques:
            return model_concepts

        for t in techniques:
            name = t.lower().replace(" ", "_")
            if len(name) < 4:
                continue
            if name not in model_concepts:
                model_concepts[name] = Concept(
                    name=name,
                    importance=1.5,
                    category=ConceptCategory.MODEL,
                    source="code",
                )
        return model_concepts

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
                    d0 = 0.3
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
                        d0 = 0.3
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
                    
                    for group in groups:
                        if projected_size >= max_size:
                            break

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

"""
evolution_agent.py

Single, simple math-driven evolution agent for a multi-agent "dream team".

- Uses an LLM to extract 10–20 key concepts for the problem.
- Tracks per-agent expertise δ_i(v) over those concepts.
- Updates δ_i(v) with a logistic learning rule based on metric improvement.
- Computes per-agent problem overlap Ω_i(P).
- Each iteration, proposes:
    - which agent to DELETE (weakest overlap)
    - a SPECIALIST child based on a strong-but-generalist agent
    - a GAP expert based on under-covered concepts

The orchestrator is responsible for:
  - actually deleting old agents and creating new ones using NewAgentSpec
  - wiring math-state snippets into agent prompts if desired
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterable, Tuple
import math

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

    def select_weakest_agent(self) -> Optional[AgentMathState]:
        """Agent with lowest Ω_i(P)."""
        if not self.agent_states:
            return None
        return min(self.agent_states.values(), key=lambda s: s.problem_overlap(self.problem))

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
        gap_threshold: float = 0.3,  # Raised from 0.2 - easier to find gaps
        specialize_overlap_threshold: float = 0.4,  # Lowered from 0.6 - easier to specialize
        specialize_gini_threshold: float = 0.5,  # Raised from 0.3 - broader candidates
        alpha: float = 0.5,
        beta: float = 0.1,
    ):
        self.llm = llm
        self.target_team_size = target_team_size
        self.gap_threshold = gap_threshold
        self.specialize_overlap_threshold = specialize_overlap_threshold
        self.specialize_gini_threshold = specialize_gini_threshold
        self.alpha = alpha
        self.beta = beta
        self.team_state: Optional[TeamMathState] = None

    # ----- LLM concept extraction -----

    def define_problem_space(
        self,
        problem_statement: str,
        target_metric: str,
        max_concepts: int = 20,
    ) -> ProblemConcepts:
        """
        Define the problem space P by extracting key concepts and their importance.
        """
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
- Format: concept_name | importance
- concept_name: short, snake_case, no spaces (e.g. gradient_boosting, microbial_growth)
- importance: integer 1, 2, or 3 (3 = very important).
- No explanations, no headings, only the raw list.
""".strip()

        raw = self.llm.generate(prompt, temperature=0.2)
        concept_weights: Dict[str, float] = {}

        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 1:
                name = parts[0]
                importance = 1.0
            else:
                name = parts[0]
                try:
                    importance = float(parts[1])
                except ValueError:
                    importance = 1.0
            if not name:
                continue
            if name in concept_weights:
                concept_weights[name] = max(concept_weights[name], importance)
            else:
                concept_weights[name] = importance
            if len(concept_weights) >= max_concepts:
                break

        if not concept_weights:
            concept_weights = {target_metric.lower(): 3.0}

        return ProblemConcepts(concept_weights=concept_weights)

    # ----- Initialization & Updates -----

    def initialize(
        self,
        agents: List[Any],
        problem_statement: str,
        target_metric: str,
        minimize_metric: bool = True,
    ) -> None:
        """
        Initialize team math state from agents + problem.
        Simple init:
          - δ_i(v) = 0.3 if concept tokens appear in agent.expertise/title, else 0.05.
        """
        problem = self.define_problem_space(problem_statement, target_metric)
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
            weakest_state = self.team_state.select_weakest_agent()
            if weakest_state is not None:
                agents_to_delete.append(weakest_state.agent_ref)

        # 4) New agent proposals (only if we're below max size OR if we're deleting someone)
        new_specs: List[NewAgentSpec] = []
        can_add = (current_size < max_size) or (len(agents_to_delete) > 0)

        if can_add:
            # 4a) specialization candidate: strong generalist
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

            # 4b) gap-based new agent
            if gaps:
                sorted_gaps = sorted(
                    gaps,
                    key=lambda c: problem.concept_weights.get(c, 0.0),
                    reverse=True,
                )
                focus = sorted_gaps[:5]
                title = "Gap-Focused Domain Expert"
                expertise = (
                    "Domain expert created to cover currently under-served concepts: "
                    + ", ".join(focus)
                )
                role = "Introduce methods, theories, and domain knowledge centered on these under-covered concepts."
                new_specs.append(
                    NewAgentSpec(
                        kind="gap",
                        title=title,
                        expertise=expertise,
                        role=role,
                        focus_concepts=focus,
                    )
                )

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

"""
Performance-Grounded Evolution for Dream Team

This module redesigns evolution to optimize for ACTUAL PERFORMANCE
instead of abstract concept coverage.

Key Changes:
1. Concepts extracted from CODE (what techniques worked), not LLM imagination
2. Agent contributions tracked via ATTRIBUTION (who proposed what)
3. Depth updates based on CAUSAL IMPACT (did this agent's idea help?)
4. New agents have PROBATION PERIOD (prove value before permanent)
5. Evolution guided by REFLECTION (PI's analysis of what's needed)

The core insight: Evolution should amplify what WORKS and prune what DOESN'T.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import re
import ast


# ============================================================================
# CONCEPT GROUNDING: Extract concepts from actual code, not imagination
# ============================================================================

class TechniqueExtractor:
    """
    Extract actual techniques used in code.
    
    Instead of asking LLM "what concepts matter?", we look at
    what the code ACTUALLY DOES.
    """
    
    # Map library/function patterns to technique concepts
    TECHNIQUE_PATTERNS = {
        # Tree-based models
        r'LGBMRegressor|LGBMClassifier|lightgbm': 'lightgbm',
        r'XGBRegressor|XGBClassifier|xgboost': 'xgboost',
        r'GradientBoosting|GBDT': 'gradient_boosting',
        r'RandomForest': 'random_forest',
        r'DecisionTree': 'decision_tree',
        
        # Linear models
        r'LinearRegression|Ridge|Lasso|ElasticNet': 'linear_models',
        r'LogisticRegression': 'logistic_regression',
        
        # Neural networks
        r'torch\.nn|nn\.Module|nn\.Linear': 'neural_networks',
        r'keras|tensorflow': 'deep_learning',
        
        # Feature engineering
        r'get_dummies|OneHotEncoder': 'one_hot_encoding',
        r'StandardScaler|MinMaxScaler|RobustScaler': 'feature_scaling',
        r'PCA|TruncatedSVD': 'dimensionality_reduction',
        r'PolynomialFeatures': 'polynomial_features',
        r'\.agg\(|groupby.*mean|groupby.*std': 'aggregation_features',
        r'shift\(|rolling\(|lag': 'lag_features',
        r'LabelEncoder|OrdinalEncoder': 'label_encoding',
        r'TargetEncoder|target.*encoding': 'target_encoding',
        
        # Cross-validation
        r'cross_val_score|KFold|StratifiedKFold': 'cross_validation',
        r'GroupKFold|TimeSeriesSplit': 'grouped_cv',
        
        # Ensembling
        r'VotingRegressor|VotingClassifier': 'voting_ensemble',
        r'StackingRegressor|StackingClassifier': 'stacking',
        r'BaggingRegressor|BaggingClassifier': 'bagging',
        
        # Hyperparameter tuning
        r'GridSearchCV|RandomizedSearchCV': 'hyperparameter_tuning',
        r'optuna|Optuna': 'bayesian_optimization',
        
        # Data handling
        r'merge\(|join\(': 'data_joining',
        r'fillna|interpolate|impute': 'missing_value_handling',
        r'clip\(|winsorize': 'outlier_handling',
    }
    
    @classmethod
    def extract_from_code(cls, code: str) -> Dict[str, float]:
        """
        Extract techniques actually used in code.
        
        Returns:
            Dict mapping technique name to confidence (0-1)
        """
        techniques = {}
        
        for pattern, technique in cls.TECHNIQUE_PATTERNS.items():
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                # More matches = higher confidence this technique is central
                confidence = min(1.0, 0.3 + len(matches) * 0.2)
                techniques[technique] = max(techniques.get(technique, 0), confidence)
        
        return techniques
    
    @classmethod
    def extract_from_imports(cls, code: str) -> Set[str]:
        """Extract imported libraries to infer capabilities."""
        imports = set()
        
        # Parse import statements
        import_pattern = r'^(?:from\s+(\w+)|import\s+(\w+))'
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            lib = match.group(1) or match.group(2)
            if lib:
                imports.add(lib.lower())
        
        return imports


@dataclass
class GroundedConcept:
    """
    A concept grounded in actual code/techniques.
    
    Unlike abstract concepts like "generalizable_product_representations",
    these map directly to implementable techniques.
    """
    name: str
    source: str  # "code", "domain", "reflection"
    importance: float
    last_used_iteration: Optional[int] = None
    success_count: int = 0
    failure_count: int = 0
    
    @property
    def effectiveness(self) -> float:
        """How often this concept led to improvement."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Unknown
        return self.success_count / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "importance": self.importance,
            "last_used_iteration": self.last_used_iteration,
            "success_count": self.success_count,
            "failure_count": self.failure_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundedConcept:
        return cls(
            name=data["name"],
            source=data["source"],
            importance=data["importance"],
            last_used_iteration=data.get("last_used_iteration"),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0)
        )


class GroundedConceptSpace:
    """
    Concept space grounded in actual techniques, not LLM imagination.
    """
    
    def __init__(self):
        self.concepts: Dict[str, GroundedConcept] = {}
        self._locked = False  # Option to freeze concept space
    
    def initialize_from_domain(self, problem_statement: str, llm) -> None:
        """
        Extract MINIMAL domain concepts (3-5 max).
        
        These are high-level problem areas, not implementation details.
        """
        prompt = f"""Identify 3-5 CORE technical areas for this problem.

Problem: {problem_statement}

Rules:
- Only concrete, actionable areas (not vague like "optimization" or "analysis")
- Each should map to a distinct expertise area
- Examples: "time_series", "tabular_data", "feature_engineering", "ensemble_methods"

Output JSON array only:
["concept1", "concept2", "concept3"]
"""
        try:
            concepts = llm.generate_json(prompt, temperature=0.3)
            if isinstance(concepts, list):
                for c in concepts[:5]:  # Max 5
                    name = c.lower().replace(" ", "_").replace("-", "_")
                    self.concepts[name] = GroundedConcept(
                        name=name,
                        source="domain",
                        importance=0.5
                    )
        except Exception:
            # Fallback: generic ML concepts
            for c in ["feature_engineering", "model_selection", "validation"]:
                self.concepts[c] = GroundedConcept(name=c, source="domain", importance=0.5)
    
    def update_from_code(self, code: str, iteration: int, metric_improved: bool) -> None:
        """
        Update concept space based on actual techniques used in code.
        
        This is the key insight: concepts come from WHAT WORKED,
        not from what we imagine might be needed.
        """
        if self._locked:
            return
        
        techniques = TechniqueExtractor.extract_from_code(code)
        
        for technique, confidence in techniques.items():
            if technique not in self.concepts:
                # New technique discovered in code
                self.concepts[technique] = GroundedConcept(
                    name=technique,
                    source="code",
                    importance=confidence
                )
            
            concept = self.concepts[technique]
            concept.last_used_iteration = iteration
            
            # Track effectiveness
            if metric_improved:
                concept.success_count += 1
                # Boost importance of techniques that work
                concept.importance = min(1.0, concept.importance + 0.1)
            else:
                concept.failure_count += 1
                # Don't reduce importance too much - might work in different context
                concept.importance = max(0.2, concept.importance - 0.05)
    
    def get_effective_concepts(self) -> List[str]:
        """Get concepts that have proven effective."""
        return [
            name for name, c in self.concepts.items()
            if c.effectiveness > 0.5 and c.success_count > 0
        ]
    
    def get_underexplored_concepts(self) -> List[str]:
        """Get concepts that haven't been tried much."""
        return [
            name for name, c in self.concepts.items()
            if c.success_count + c.failure_count < 2
        ]
    
    def lock(self):
        """Freeze concept space (useful when things are working)."""
        self._locked = True
    
    def unlock(self):
        """Unfreeze concept space."""
        self._locked = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "locked": self._locked
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundedConceptSpace:
        space = cls()
        space._locked = data.get("locked", False)
        if "concepts" in data:
            space.concepts = {
                k: GroundedConcept.from_dict(v) 
                for k, v in data["concepts"].items()
            }
        return space


# ============================================================================
# ATTRIBUTION: Track which agent actually contributed to improvements
# ============================================================================

@dataclass
class Contribution:
    """Track an agent's contribution to an iteration."""
    agent_title: str
    iteration: int
    role: str  # "proposer", "implementer", "reviewer"
    techniques_advocated: List[str]
    metric_before: Optional[float]
    metric_after: Optional[float]
    
    @property
    def improvement(self) -> Optional[float]:
        if self.metric_before is None or self.metric_after is None:
            return None
        return self.metric_before - self.metric_after  # Positive = improved (for minimize)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_title": self.agent_title,
            "iteration": self.iteration,
            "role": self.role,
            "techniques_advocated": self.techniques_advocated,
            "metric_before": self.metric_before,
            "metric_after": self.metric_after
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Contribution:
        return cls(
            agent_title=data["agent_title"],
            iteration=data["iteration"],
            role=data["role"],
            techniques_advocated=data.get("techniques_advocated", []),
            metric_before=data.get("metric_before"),
            metric_after=data.get("metric_after")
        )


class ContributionTracker:
    """
    Track which agents contributed to which outcomes.
    
    This enables CAUSAL attribution: who actually helped vs who just participated.
    """
    
    def __init__(self):
        self.contributions: List[Contribution] = []
        self.agent_scores: Dict[str, float] = {}  # Running effectiveness score
    
    def record_contribution(
        self,
        agent_title: str,
        iteration: int,
        role: str,
        techniques_advocated: List[str],
        metric_before: Optional[float] = None
    ) -> Contribution:
        """Record that an agent contributed to an iteration."""
        contrib = Contribution(
            agent_title=agent_title,
            iteration=iteration,
            role=role,
            techniques_advocated=techniques_advocated,
            metric_before=metric_before,
            metric_after=None  # Filled in later
        )
        self.contributions.append(contrib)
        return contrib
    
    def record_outcome(self, iteration: int, metric_after: float):
        """Record the outcome of an iteration."""
        for contrib in self.contributions:
            if contrib.iteration == iteration and contrib.metric_after is None:
                contrib.metric_after = metric_after
                
                # Update agent score
                if contrib.improvement is not None:
                    self._update_agent_score(contrib.agent_title, contrib.improvement)
    
    def _update_agent_score(self, agent_title: str, improvement: float):
        """Update running effectiveness score for agent."""
        current = self.agent_scores.get(agent_title, 0.5)
        # Exponential moving average
        alpha = 0.3
        # Normalize improvement to [-1, 1] range roughly
        normalized = max(-1, min(1, improvement))
        new_score = current * (1 - alpha) + (0.5 + normalized * 0.5) * alpha
        self.agent_scores[agent_title] = new_score
    
    def get_agent_effectiveness(self, agent_title: str) -> float:
        """Get effectiveness score for an agent (0-1, higher = better)."""
        return self.agent_scores.get(agent_title, 0.5)
    
    def get_top_contributors(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get the most effective agents."""
        sorted_agents = sorted(
            self.agent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_agents[:n]
    
    def get_underperformers(self, threshold: float = 0.4) -> List[str]:
        """Get agents who consistently don't help."""
        return [
            agent for agent, score in self.agent_scores.items()
            if score < threshold
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contributions": [c.to_dict() for c in self.contributions],
            "agent_scores": self.agent_scores
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContributionTracker:
        tracker = cls()
        tracker.agent_scores = data.get("agent_scores", {})
        if "contributions" in data:
            tracker.contributions = [
                Contribution.from_dict(c) for c in data["contributions"]
            ]
        return tracker


# ============================================================================
# PROBATION: New agents must prove their value
# ============================================================================

class AgentStatus(Enum):
    PERMANENT = "permanent"      # Proven value, won't be auto-removed
    PROBATION = "probation"      # Must prove value within N iterations
    CANDIDATE = "candidate"      # Proposed but not yet added


@dataclass
class ProbationaryAgent:
    """
    New agents go through probation before becoming permanent.
    
    This prevents the "add specialist, metric gets worse, keep specialist anyway" problem.
    """
    agent: Any  # Agent object
    status: AgentStatus
    added_iteration: int
    metric_at_addition: float
    metrics_during_probation: List[Tuple[int, float]] = field(default_factory=list)
    probation_length: int = 3  # Iterations to prove value
    contributions: int = 0  # How many times they contributed to discussion
    
    def record_metric(self, iteration: int, metric: float):
        """Record metric during probation."""
        self.metrics_during_probation.append((iteration, metric))
    
    def record_contribution(self):
        """Record that agent contributed to team discussion."""
        self.contributions += 1
    
    def evaluate(self, minimize: bool = True) -> Tuple[bool, str]:
        """
        Evaluate whether agent should become permanent.
        
        Returns:
            (should_keep, reason)
        """
        if len(self.metrics_during_probation) < self.probation_length:
            return True, "Still in probation period"
        
        # Check 1: Did metrics improve on average?
        recent_metrics = [m for _, m in self.metrics_during_probation[-self.probation_length:]]
        avg_metric = sum(recent_metrics) / len(recent_metrics)
        
        if minimize:
            improved = avg_metric < self.metric_at_addition
        else:
            improved = avg_metric > self.metric_at_addition
        
        if not improved:
            return False, f"Metrics did not improve (before: {self.metric_at_addition:.4f}, avg during: {avg_metric:.4f})"
        
        # Check 2: Did agent actually contribute?
        if self.contributions == 0:
            return False, "Agent never contributed to discussions"
        
        return True, f"Metrics improved and agent contributed {self.contributions} times"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_title": getattr(self.agent, 'title', str(self.agent)), # Store title only
            "status": self.status.value,
            "added_iteration": self.added_iteration,
            "metric_at_addition": self.metric_at_addition,
            "metrics_during_probation": self.metrics_during_probation,
            "probation_length": self.probation_length,
            "contributions": self.contributions
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], agent_map: Dict[str, Any]) -> ProbationaryAgent:
        # We need to reconnect to the actual agent object
        title = data["agent_title"]
        agent = agent_map.get(title) # Might be None if agent was removed, but shouldn't happen for active probation
        
        return cls(
            agent=agent,
            status=AgentStatus(data["status"]),
            added_iteration=data["added_iteration"],
            metric_at_addition=data["metric_at_addition"],
            metrics_during_probation=data.get("metrics_during_probation", []),
            probation_length=data.get("probation_length", 3),
            contributions=data.get("contributions", 0)
        )


class ProbationManager:
    """Manage probationary agents."""
    
    def __init__(self, probation_length: int = 3):
        self.probation_length = probation_length
        self.agents: Dict[str, ProbationaryAgent] = {}
    
    def add_to_probation(
        self,
        agent: Any,
        iteration: int,
        current_metric: float
    ) -> ProbationaryAgent:
        """Add new agent to probation."""
        title = getattr(agent, 'title', str(agent))
        
        probationary = ProbationaryAgent(
            agent=agent,
            status=AgentStatus.PROBATION,
            added_iteration=iteration,
            metric_at_addition=current_metric,
            probation_length=self.probation_length
        )
        
        self.agents[title] = probationary
        return probationary
    
    def make_permanent(self, agent_title: str):
        """Graduate agent from probation to permanent."""
        if agent_title in self.agents:
            self.agents[agent_title].status = AgentStatus.PERMANENT
    
    def record_iteration(
        self,
        iteration: int,
        metric: float,
        contributing_agents: List[str]
    ):
        """Record iteration results for all probationary agents."""
        for title, prob_agent in self.agents.items():
            if prob_agent.status == AgentStatus.PROBATION:
                prob_agent.record_metric(iteration, metric)
                if title in contributing_agents:
                    prob_agent.record_contribution()
    
    def evaluate_all(self, minimize: bool = True) -> List[Tuple[str, bool, str]]:
        """
        Evaluate all probationary agents.
        
        Returns:
            List of (agent_title, should_keep, reason)
        """
        results = []
        for title, prob_agent in self.agents.items():
            if prob_agent.status == AgentStatus.PROBATION:
                should_keep, reason = prob_agent.evaluate(minimize)
                results.append((title, should_keep, reason))
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probation_length": self.probation_length,
            "agents": {k: v.to_dict() for k, v in self.agents.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], agent_map: Dict[str, Any]) -> ProbationManager:
        manager = cls(probation_length=data.get("probation_length", 3))
        if "agents" in data:
            manager.agents = {
                k: ProbationaryAgent.from_dict(v, agent_map)
                for k, v in data["agents"].items()
            }
        return manager


# ============================================================================
# REFLECTION-GUIDED EVOLUTION: Use PI's insights to guide evolution
# ============================================================================

class ReflectionEvolutionGuide:
    """
    Use PI's reflection to guide evolution decisions.
    
    Instead of detecting abstract "gaps", we use the PI's actual analysis
    of what went wrong and what's needed.
    """
    
    # Keywords that suggest expertise needs
    EXPERTISE_SIGNALS = {
        'feature': ['feature_engineering'],
        'encoding': ['categorical_encoding', 'feature_engineering'],
        'tree': ['tree_models', 'gradient_boosting'],
        'neural': ['deep_learning', 'neural_networks'],
        'ensemble': ['ensemble_methods'],
        'overfit': ['regularization', 'cross_validation'],
        'variance': ['ensemble_methods', 'regularization'],
        'bias': ['feature_engineering', 'model_complexity'],
        'scale': ['feature_scaling', 'normalization'],
        'missing': ['missing_value_handling'],
        'outlier': ['outlier_handling', 'robust_methods'],
        'time': ['time_series', 'temporal_features'],
        'lag': ['lag_features', 'time_series'],
        'target leak': ['data_leakage_prevention'],
        'cv': ['cross_validation'],
        'hyperparameter': ['hyperparameter_tuning'],
    }
    
    @classmethod
    def extract_expertise_needs(cls, reflection_text: str) -> List[str]:
        """
        Extract what expertise is needed based on reflection.
        
        This is more grounded than abstract concept gaps because it's
        based on the PI's actual analysis of what went wrong.
        """
        needs = set()
        reflection_lower = reflection_text.lower()
        
        for keyword, expertise_list in cls.EXPERTISE_SIGNALS.items():
            if keyword in reflection_lower:
                needs.update(expertise_list)
        
        return list(needs)
    
    @classmethod
    def extract_dead_ends(cls, reflection_text: str, dead_ends: List[str]) -> List[str]:
        """
        Extract what approaches to avoid.
        
        Agents specializing in dead-end approaches should be deprioritized.
        """
        avoid = []
        
        for dead_end in dead_ends:
            dead_end_lower = dead_end.lower()
            for keyword, expertise_list in cls.EXPERTISE_SIGNALS.items():
                if keyword in dead_end_lower:
                    avoid.extend(expertise_list)
        
        return list(set(avoid))


# ============================================================================
# GROUNDED EVOLUTION AGENT: Putting it all together
# ============================================================================

@dataclass
class GroundedEvolutionDecision:
    """Evolution decision grounded in actual performance."""
    agents_to_delete: List[Any]
    new_agent_specs: List[Dict[str, str]]  # Specs for new agents
    agents_to_graduate: List[str]  # Probationary → Permanent
    reasoning: str
    confidence: float  # How confident we are this will help
    debug_info: Dict[str, Any] = field(default_factory=dict) # Added for compatibility

    @property
    def quality(self) -> float:
        """Alias for confidence to match legacy interface."""
        return self.confidence
    
    @property
    def agents_to_remove(self) -> List[Any]:
        """Alias for agents_to_delete to match legacy interface."""
        return self.agents_to_delete


class GroundedEvolutionAgent:
    """
    Evolution agent that makes decisions based on ACTUAL PERFORMANCE,
    not abstract concept coverage.
    
    Key principles:
    1. Concepts come from code (what techniques are used)
    2. Credit goes to agents who contributed to improvements
    3. New agents must prove value in probation
    4. Evolution is guided by PI's reflection, not abstract gaps
    """
    
    def __init__(
        self,
        llm: Any,
        min_team_size: int = 3,
        max_team_size: int = 5,
        probation_length: int = 3,
        evolution_cooldown: int = 2,
    ):
        self.llm = llm
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size
        self.evolution_cooldown = evolution_cooldown
        
        # Core components
        self.concept_space = GroundedConceptSpace()
        self.contribution_tracker = ContributionTracker()
        self.probation_manager = ProbationManager(probation_length)
        
        # State
        self.metric_history: List[Tuple[int, float]] = []
        self.minimize_metric: bool = True
        self.last_evolution_iteration: int = 0
        self._initialized = False
    
    def initialize(
        self,
        problem_statement: str,
        agents: List[Any],
        minimize_metric: bool = True
    ):
        """Initialize with problem and initial team."""
        self.minimize_metric = minimize_metric
        
        # Initialize concept space from domain (minimal, high-level)
        self.concept_space.initialize_from_domain(problem_statement, self.llm)
        
        # Mark initial agents as permanent (they've been chosen by bootstrap)
        for agent in agents:
            title = getattr(agent, 'title', str(agent))
            self.contribution_tracker.agent_scores[title] = 0.5  # Neutral starting score
        
        self._initialized = True
    
    def record_iteration(
        self,
        iteration: int,
        metric: float,
        code: str,
        approach: str,
        contributing_agents: List[str],
        reflection_text: str = "",
        reflection_obj: Any = None,
    ):
        """
        Record iteration results and update all tracking.
        
        This is called AFTER each iteration completes.
        """
        # Track metric
        prev_metric = self.metric_history[-1][1] if self.metric_history else None
        self.metric_history.append((iteration, metric))
        
        # Determine if improved
        improved = False
        if prev_metric is not None:
            if self.minimize_metric:
                improved = metric < prev_metric
            else:
                improved = metric > prev_metric
        
        # Update concept space from actual code
        self.concept_space.update_from_code(code, iteration, improved)
        
        # Record outcome for contribution tracking
        self.contribution_tracker.record_outcome(iteration, metric)
        
        # Update probationary agents
        self.probation_manager.record_iteration(iteration, metric, contributing_agents)
    
    def record_agent_contribution(
        self,
        agent_title: str,
        iteration: int,
        role: str,
        techniques_mentioned: List[str]
    ):
        """Record that an agent contributed to an iteration."""
        prev_metric = self.metric_history[-1][1] if self.metric_history else None
        self.contribution_tracker.record_contribution(
            agent_title=agent_title,
            iteration=iteration,
            role=role,
            techniques_advocated=techniques_mentioned,
            metric_before=prev_metric
        )
    
    def step(
        self,
        current_iteration: int,
        current_metric: float,
        current_team: List[Any],
        reflection_text: str = "",
        reflection_obj: Any = None,
    ) -> GroundedEvolutionDecision:
        """
        Make evolution decision based on actual performance.
        """
        # Check cooldown
        if current_iteration - self.last_evolution_iteration < self.evolution_cooldown:
            return GroundedEvolutionDecision(
                agents_to_delete=[],
                new_agent_specs=[],
                agents_to_graduate=[],
                reasoning=f"Cooldown: {self.evolution_cooldown - (current_iteration - self.last_evolution_iteration)} iterations remaining",
                confidence=1.0
            )
        
        # Check if things are improving - if so, don't mess with it
        if self._is_improving():
            return GroundedEvolutionDecision(
                agents_to_delete=[],
                new_agent_specs=[],
                agents_to_graduate=[],
                reasoning="Performance is improving - no changes needed",
                confidence=1.0
            )
        
        agents_to_remove = []
        agents_to_add = []
        agents_to_graduate = []
        reasoning_parts = []
        
        # 1. Evaluate probationary agents
        probation_results = self.probation_manager.evaluate_all(self.minimize_metric)
        for title, should_keep, reason in probation_results:
            if should_keep:
                agents_to_graduate.append(title)
                reasoning_parts.append(f"Graduate {title}: {reason}")
            else:
                # Find agent object
                for agent in current_team:
                    if getattr(agent, 'title', '') == title:
                        agents_to_remove.append(agent)
                        reasoning_parts.append(f"Remove {title}: {reason}")
                        break
        
        # 2. Check for underperforming permanent agents (only if team > min)
        if len(current_team) - len(agents_to_remove) > self.min_team_size:
            underperformers = self.contribution_tracker.get_underperformers(threshold=0.35)
            for title in underperformers:
                # Don't remove if already being removed
                if any(getattr(a, 'title', '') == title for a in agents_to_remove):
                    continue
                # Don't remove below min team size
                if len(current_team) - len(agents_to_remove) <= self.min_team_size:
                    break
                # Find and mark for removal
                for agent in current_team:
                    if getattr(agent, 'title', '') == title:
                        agents_to_remove.append(agent)
                        reasoning_parts.append(f"Remove {title}: Consistently underperforming (score: {self.contribution_tracker.get_agent_effectiveness(title):.2f})")
                        break
        
        # 3. Consider adding new agent only if:
        #    - Team is below max size
        #    - Performance has stagnated (not improving)
        #    - Reflection suggests specific expertise need
        projected_size = len(current_team) - len(agents_to_remove)
        
        if projected_size < self.max_team_size and self._is_stagnated():
            # Use reflection to guide what's needed (not abstract gaps)
            if reflection_text:
                needed_expertise = ReflectionEvolutionGuide.extract_expertise_needs(reflection_text)
                
                # Filter to expertise not already well-covered
                top_contributors = self.contribution_tracker.get_top_contributors(3)
                covered_expertise = set()
                for agent in current_team:
                    if any(getattr(agent, 'title', '') == title for title, _ in top_contributors):
                        # Extract expertise from agent
                        expertise = getattr(agent, 'expertise', '').lower()
                        covered_expertise.update(expertise.split(','))
                
                uncovered_needs = [e for e in needed_expertise if e not in covered_expertise]
                
                if uncovered_needs and projected_size < self.max_team_size:
                    # Create ONE specialist for the most needed expertise
                    focus = uncovered_needs[0]
                    agents_to_add.append({
                        'title': f'{focus.replace("_", " ").title()} Specialist',
                        'expertise': f'Specialized in {focus.replace("_", " ")}',
                        'role': f'Provide expertise in {focus.replace("_", " ")}',
                        'focus_concepts': [focus],
                    })
                    reasoning_parts.append(f"Add specialist for '{focus}': Identified need from reflection")
        
        # Calculate confidence
        confidence = 0.5
        if agents_to_remove and not agents_to_add:
            confidence = 0.7  # Pruning is usually safe
        elif agents_to_add and not agents_to_remove:
            confidence = 0.4  # Adding is risky
        elif agents_to_remove and agents_to_add:
            confidence = 0.5  # Mixed changes
        
        if agents_to_remove or agents_to_add or agents_to_graduate:
            self.last_evolution_iteration = current_iteration
        
        return GroundedEvolutionDecision(
            agents_to_delete=agents_to_remove,
            new_agent_specs=agents_to_add,
            agents_to_graduate=agents_to_graduate,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "No changes needed",
            confidence=confidence,
            debug_info={
                "team_size": len(current_team),
                "min_size": self.min_team_size,
                "max_size": self.max_team_size,
                "reasoning": reasoning_parts
            }
        )
    
    def _is_improving(self, window: int = 2) -> bool:
        """Check if performance is improving over recent iterations."""
        if len(self.metric_history) < window + 1:
            return False
        
        recent = [m for _, m in self.metric_history[-window:]]
        older = self.metric_history[-(window + 1)][1]
        
        avg_recent = sum(recent) / len(recent)
        
        if self.minimize_metric:
            return avg_recent < older * 0.99  # At least 1% improvement
        else:
            return avg_recent > older * 1.01
    
    def _is_stagnated(self, window: int = 3, threshold: float = 0.02) -> bool:
        """Check if performance has stagnated."""
        if len(self.metric_history) < window:
            return False
        
        recent = [m for _, m in self.metric_history[-window:]]
        variation = (max(recent) - min(recent)) / max(abs(recent[0]), 1e-6)
        
        return variation < threshold
    
    def get_coverage_and_gaps(self) -> Tuple[Dict[str, float], List[str]]:
        """
        Get coverage and gaps for ContextBuilder compatibility.
        
        Returns:
            (coverage_dict, gaps_list)
        """
        coverage = {c.name: c.importance for c in self.concept_space.concepts.values()}
        gaps = self.concept_space.get_underexplored_concepts()
        return coverage, gaps

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state."""
        return {
            "min_team_size": self.min_team_size,
            "max_team_size": self.max_team_size,
            "evolution_cooldown": self.evolution_cooldown,
            "concept_space": self.concept_space.to_dict(),
            "contribution_tracker": self.contribution_tracker.to_dict(),
            "probation_manager": self.probation_manager.to_dict(),
            "metric_history": self.metric_history,
            "minimize_metric": self.minimize_metric,
            "last_evolution_iteration": self.last_evolution_iteration,
            "initialized": self._initialized
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], llm: Any, agent_map: Dict[str, Any]) -> GroundedEvolutionAgent:
        """Restore state."""
        agent = cls(
            llm=llm,
            min_team_size=data.get("min_team_size", 3),
            max_team_size=data.get("max_team_size", 5),
            probation_length=data.get("probation_manager", {}).get("probation_length", 3),
            evolution_cooldown=data.get("evolution_cooldown", 2)
        )
        
        agent.metric_history = data.get("metric_history", [])
        agent.minimize_metric = data.get("minimize_metric", True)
        agent.last_evolution_iteration = data.get("last_evolution_iteration", 0)
        agent._initialized = data.get("initialized", False)
        
        if "concept_space" in data:
            agent.concept_space = GroundedConceptSpace.from_dict(data["concept_space"])
            
        if "contribution_tracker" in data:
            agent.contribution_tracker = ContributionTracker.from_dict(data["contribution_tracker"])
            
        if "probation_manager" in data:
            agent.probation_manager = ProbationManager.from_dict(data["probation_manager"], agent_map)
            
        return agent

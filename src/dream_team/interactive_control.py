"""
Interactive Control Module for Dream Team Framework.

Provides human-in-the-loop oversight at critical decision points:
1. Evolution decisions (add/remove agents)
2. Post-reflection steering
3. Concept space changes
4. Performance-based guardrails

This prevents the "runaway evolution" problem where the system
optimizes for abstract metrics instead of actual performance.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ControlPoint(Enum):
    """Decision points where user can intervene."""
    PRE_EVOLUTION = "pre_evolution"      # Before any team changes
    POST_REFLECTION = "post_reflection"  # After PI reflects, before next iteration
    CONCEPT_CHANGE = "concept_change"    # When concept space is refined



@dataclass
class EvolutionProposal:
    """Structured proposal for team evolution."""
    agents_to_remove: List[str]
    agents_to_add: List[Dict[str, str]]  # [{title, expertise, role, reason}]
    coverage_gaps: List[str]
    current_team: List[str]
    reason: str
    confidence: float  # 0-1, how confident the system is this will help


@dataclass 
class PerformanceGuard:
    """
    Guards against runaway evolution by tracking performance.
    
    Rules:
    - Don't evolve if metric improved this iteration
    - Don't evolve if we evolved recently and haven't seen results
    - Rollback if performance degraded significantly after evolution
    """
    best_metric: Optional[float] = None
    best_metric_iteration: int = 0
    last_evolution_iteration: int = 0
    evolution_cooldown: int = 2  # Min iterations between evolutions
    performance_after_evolution: List[float] = field(default_factory=list)
    minimize_metric: bool = True
    
    def should_allow_evolution(
        self, 
        current_iteration: int,
        current_metric: Optional[float],
        last_metric: Optional[float]
    ) -> Tuple[bool, str]:
        """
        Determine if evolution should be allowed.
        
        Returns:
            (allowed, reason)
        """
        # Rule 1: Don't evolve if we just evolved
        iterations_since_evolution = current_iteration - self.last_evolution_iteration
        if iterations_since_evolution < self.evolution_cooldown:
            return False, f"Cooldown: {self.evolution_cooldown - iterations_since_evolution} iterations until evolution allowed"
        
        # Rule 2: Don't evolve if this iteration improved
        if current_metric is not None and last_metric is not None:
            improved = (current_metric < last_metric) if self.minimize_metric else (current_metric > last_metric)
            if improved:
                return False, "Performance improved this iteration - no evolution needed"
        
        # Rule 3: Check if we're doing worse than before evolution
        if self.performance_after_evolution:
            pre_evolution_metric = self.performance_after_evolution[0] if self.performance_after_evolution else None
            if pre_evolution_metric is not None and current_metric is not None:
                degraded = (current_metric > pre_evolution_metric * 1.1) if self.minimize_metric else (current_metric < pre_evolution_metric * 0.9)
                if degraded:
                    return True, "⚠️ Performance degraded >10% since last evolution - consider rollback"
        
        # Rule 4: Only evolve if stagnated for 2+ iterations
        if self.best_metric is not None and current_metric is not None:
            at_best = abs(current_metric - self.best_metric) < 0.01 * abs(self.best_metric)
            iterations_since_best = current_iteration - self.best_metric_iteration
            if at_best or iterations_since_best < 2:
                return False, "Performance not stagnated long enough"
        
        return True, "Evolution allowed - performance stagnated"
    
    def record_evolution(self, iteration: int, pre_metric: Optional[float]):
        """Record that evolution happened."""
        self.last_evolution_iteration = iteration
        self.performance_after_evolution = [pre_metric] if pre_metric else []
    
    def record_metric(self, iteration: int, metric: float):
        """Record a metric value."""
        if self.last_evolution_iteration > 0:
            self.performance_after_evolution.append(metric)
        
        if self.best_metric is None:
            self.best_metric = metric
            self.best_metric_iteration = iteration
        elif (self.minimize_metric and metric < self.best_metric) or \
             (not self.minimize_metric and metric > self.best_metric):
            self.best_metric = metric
            self.best_metric_iteration = iteration


class InteractiveController:
    """
    Human-in-the-loop controller for Dream Team.
    
    Provides prompts at critical decision points and collects user feedback.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        auto_approve_improvements: bool = True,  # Skip prompts when things improve
        require_approval_for: List[ControlPoint] = None
    ):
        self.enabled = enabled
        self.auto_approve_improvements = auto_approve_improvements
        self.require_approval_for = require_approval_for or [
            ControlPoint.PRE_EVOLUTION,
            ControlPoint.POST_REFLECTION
        ]
        self.performance_guard = PerformanceGuard()
        self.user_notes: List[Dict[str, Any]] = []  # Accumulated user feedback
        
    def _prompt_user(self, message: str, options: List[str] = None) -> str:
        """Display prompt and get user input."""
        print("\n" + "=" * 60)
        print("🎮 HUMAN INPUT REQUESTED")
        print("=" * 60)
        print(message)
        
        if options:
            print("\nOptions:")
            for i, opt in enumerate(options):
                print(f"  [{i+1}] {opt}")
            print(f"  [0] Skip / Continue with default")
        
        try:
            response = input("\nYour choice (or free text): ").strip()
        except EOFError:
            print("   (No input available, continuing with default)")
            return "0"
        
        return response
    
    def check_evolution_proposal(
        self,
        proposal: EvolutionProposal,
        current_iteration: int,
        current_metric: Optional[float],
        last_metric: Optional[float],
        target_metric: str
    ) -> Tuple[bool, Optional[str], Optional[List[Dict]]]:
        """
        Review evolution proposal with user.
        
        Returns:
            (approved, user_feedback, modified_proposal)
            - approved: Whether to proceed
            - user_feedback: Any notes from user to inject into context
            - modified_proposal: If user modified the proposal
        """
        if not self.enabled:
            return True, None, None
        
        # First check performance guard
        allowed, guard_reason = self.performance_guard.should_allow_evolution(
            current_iteration, current_metric, last_metric
        )
        
        if not allowed and self.auto_approve_improvements:
            print(f"\n🛡️ Evolution Guard: {guard_reason}")
            print("   Skipping evolution (auto-approve mode)")
            return False, None, None
        
        # Build proposal summary
        summary = f"""
📊 EVOLUTION PROPOSAL (Iteration {current_iteration})

Current Performance:
  • {target_metric}: {current_metric:.4f if current_metric else 'N/A'}
  • Best so far: {self.performance_guard.best_metric:.4f if self.performance_guard.best_metric else 'N/A'}

Guard Status: {guard_reason}

Current Team ({len(proposal.current_team)} members):
{chr(10).join(f'  • {name}' for name in proposal.current_team)}

Proposed Changes:
"""
        
        if proposal.agents_to_remove:
            summary += f"\n🗑️ REMOVE ({len(proposal.agents_to_remove)}):\n"
            for name in proposal.agents_to_remove:
                summary += f"  • {name}\n"
        
        if proposal.agents_to_add:
            summary += f"\n✨ ADD ({len(proposal.agents_to_add)}):\n"
            for agent in proposal.agents_to_add:
                summary += f"  • {agent.get('title', 'Unknown')}\n"
                summary += f"    Expertise: {agent.get('expertise', 'N/A')[:80]}...\n"
                summary += f"    Reason: {agent.get('reason', 'Coverage gap')}\n"
        
        if proposal.coverage_gaps:
            summary += f"\nCoverage Gaps Detected:\n"
            for gap in proposal.coverage_gaps[:5]:
                summary += f"  • {gap}\n"
        
        summary += f"\nConfidence: {proposal.confidence:.0%}"
        summary += f"\nReason: {proposal.reason}"
        
        # Prompt user
        response = self._prompt_user(
            summary,
            options=[
                "Approve all changes",
                "Approve additions only (keep all current agents)", 
                "Approve removals only (don't add new agents)",
                "Reject all changes (keep team as-is)",
                "Provide feedback and reject"
            ]
        )
        
        # Parse response
        if response == "1":
            self.performance_guard.record_evolution(current_iteration, current_metric)
            return True, None, None
        elif response == "2":
            # Only additions
            modified = {"agents_to_add": proposal.agents_to_add, "agents_to_remove": []}
            self.performance_guard.record_evolution(current_iteration, current_metric)
            return True, None, modified
        elif response == "3":
            # Only removals
            modified = {"agents_to_add": [], "agents_to_remove": proposal.agents_to_remove}
            self.performance_guard.record_evolution(current_iteration, current_metric)
            return True, None, modified
        elif response == "4" or response == "0":
            return False, None, None
        elif response == "5":
            feedback = input("Enter your feedback: ").strip()
            self.user_notes.append({
                "iteration": current_iteration,
                "type": "evolution_rejection",
                "feedback": feedback
            })
            return False, feedback, None
        else:
            # Free text response - treat as feedback and reject
            if response:
                self.user_notes.append({
                    "iteration": current_iteration,
                    "type": "evolution_feedback",
                    "feedback": response
                })
            return False, response if response else None, None
    
    def check_reflection(
        self,
        reflection_text: str,
        iteration: int,
        metrics: Dict[str, float],
        target_metric: str
    ) -> Optional[str]:
        """
        Show reflection and get optional user steering.
        
        Returns:
            Optional user feedback to inject into next planning meeting
        """
        if not self.enabled or ControlPoint.POST_REFLECTION not in self.require_approval_for:
            return None
        
        # Check if we should auto-skip (performance improved)
        current = metrics.get(target_metric)
        if current and self.performance_guard.best_metric:
            improved = (current <= self.performance_guard.best_metric) if self.performance_guard.minimize_metric else (current >= self.performance_guard.best_metric)
            if improved and self.auto_approve_improvements:
                print("\n✅ Performance improved - continuing automatically")
                return None
        
        # Show reflection summary (truncated)
        summary = f"""
🤔 PI REFLECTION (Iteration {iteration})

Metrics: {', '.join(f'{k}={v:.4f}' for k,v in metrics.items())}
Best {target_metric}: {self.performance_guard.best_metric:.4f if self.performance_guard.best_metric else 'N/A'}

--- Reflection (truncated) ---
{reflection_text[:1500]}{'...' if len(reflection_text) > 1500 else ''}
--- End ---

You can provide steering feedback for the next iteration.
This will be included in the team planning meeting context.
"""
        
        response = self._prompt_user(
            summary,
            options=[
                "Continue without feedback",
                "Provide steering feedback",
                "Force specific approach next iteration"
            ]
        )
        
        if response == "1" or response == "0":
            return None
        elif response == "2":
            feedback = input("Enter steering feedback: ").strip()
            if feedback:
                self.user_notes.append({
                    "iteration": iteration,
                    "type": "steering",
                    "feedback": feedback
                })
                return feedback
        elif response == "3":
            approach = input("Describe the approach to try: ").strip()
            if approach:
                self.user_notes.append({
                    "iteration": iteration,
                    "type": "forced_approach",
                    "feedback": approach
                })
                return f"USER DIRECTIVE: The next iteration MUST try this approach: {approach}"
        else:
            # Free text
            if response:
                self.user_notes.append({
                    "iteration": iteration,
                    "type": "steering",
                    "feedback": response
                })
                return response
        
        return None
    
    def check_concept_changes(
        self,
        old_concepts: List[str],
        new_concepts: List[str],
        iteration: int
    ) -> Tuple[bool, Optional[List[str]]]:
        """
        Review concept space changes.
        
        Returns:
            (approved, modified_concepts)
        """
        if not self.enabled or ControlPoint.CONCEPT_CHANGE not in self.require_approval_for:
            return True, None
        
        added = set(new_concepts) - set(old_concepts)
        removed = set(old_concepts) - set(new_concepts)
        
        if not added and not removed:
            return True, None
        
        summary = f"""
🧠 CONCEPT SPACE CHANGE (Iteration {iteration})

Current concepts: {', '.join(old_concepts)}

Changes:
  Added: {', '.join(added) if added else 'None'}
  Removed: {', '.join(removed) if removed else 'None'}

New concepts: {', '.join(new_concepts)}
"""
        
        response = self._prompt_user(
            summary,
            options=[
                "Approve changes",
                "Keep old concepts (reject changes)",
                "Manually specify concepts"
            ]
        )
        
        if response == "1":
            return True, None
        elif response == "2" or response == "0":
            return False, old_concepts
        elif response == "3":
            concepts_input = input("Enter concepts (comma-separated): ").strip()
            if concepts_input:
                manual_concepts = [c.strip() for c in concepts_input.split(",")]
                return True, manual_concepts
        
        return True, None
    

    
    def get_accumulated_feedback(self) -> str:
        """
        Get all accumulated user feedback for injection into context.
        """
        if not self.user_notes:
            return ""
        
        feedback_str = "\n## User Feedback (PRIORITY)\n\n"
        for note in self.user_notes[-3:]:  # Last 3 notes
            feedback_str += f"Iteration {note['iteration']} ({note['type']}): {note['feedback']}\n"
        
        return feedback_str


# ============================================================================
# Integration helpers
# ============================================================================

def create_evolution_proposal(
    evolution_decision,  # EvolutionDecision or GroundedEvolutionDecision
    current_team: List,  # List of Agent objects
) -> EvolutionProposal:
    """Convert EvolutionDecision to EvolutionProposal for user review."""
    
    agents_to_add = []
    for spec in evolution_decision.new_agent_specs:
        # Handle both object (legacy) and dict (grounded) specs
        if isinstance(spec, dict):
            agents_to_add.append({
                "title": spec.get("title", "Unknown"),
                "expertise": spec.get("expertise", "N/A"),
                "role": spec.get("role", "N/A"),
                "reason": f"Focus: {', '.join(spec.get('focus_concepts', []))}"
            })
        else:
            agents_to_add.append({
                "title": spec.title,
                "expertise": spec.expertise,
                "role": spec.role,
                "reason": f"Gap in: {', '.join(spec.focus_concepts)}"
            })
    
    # Handle graduations (if present)
    if hasattr(evolution_decision, "agents_to_graduate") and evolution_decision.agents_to_graduate:
        for title in evolution_decision.agents_to_graduate:
            # We treat graduations as "additions" to permanent status in the UI for now, 
            # or just list them in the reason since they aren't new agents.
            # Better: Append to reason.
            pass

    agents_to_remove = [
        getattr(a, 'title', str(a)) 
        for a in evolution_decision.agents_to_delete
    ]
    
    current_team_names = [
        getattr(a, 'title', str(a)) 
        for a in current_team
    ]
    
    coverage_gaps = evolution_decision.debug_info.get('gaps', [])
    
    # Compute confidence based on quality and gap severity
    quality = getattr(evolution_decision, 'quality', getattr(evolution_decision, 'confidence', 0.5))
    num_gaps = len(coverage_gaps)
    confidence = min(0.9, 0.5 + (quality - 0.5) * 0.4 - num_gaps * 0.05)
    
    reason = getattr(evolution_decision, 'reasoning', f"Quality={quality:.2f}, {num_gaps} coverage gaps")
    
    return EvolutionProposal(
        agents_to_remove=agents_to_remove,
        agents_to_add=agents_to_add,
        coverage_gaps=coverage_gaps,
        current_team=current_team_names,
        reason=reason,
        confidence=max(0.1, confidence)
    )

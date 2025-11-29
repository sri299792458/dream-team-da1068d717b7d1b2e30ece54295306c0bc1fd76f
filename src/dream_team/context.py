"""
Context management for robust iteration tracking.

Implements Reflexion-based learning (Shinn et al.):
- After each iteration, explicitly reflect on what happened
- Store structured learnings in memory
- Use past reflections to guide future iterations
"""

from dataclasses import dataclass, field
from typing import List, Optional

# Note: ExperimentState will be enhanced in orchestrator.py
# to include IterationRecord and EventStore references



@dataclass
class Reflection:
    """
    Structured reflection on an iteration (based on Reflexion paper by Shinn et al.)

    After each iteration, the team lead reflects on what happened
    and extracts structured learnings.

    Workflow:
        execute → metrics → REFLECT → store learnings → next iteration
    """
    iteration: int
    what_worked: List[str] = field(default_factory=list)
    what_failed: List[str] = field(default_factory=list)
    why_failed: str = ""
    suggestions: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)

    @classmethod
    def from_text(cls, iteration: int, reflection_text: str) -> "Reflection":
        """
        Parse reflection text into structured format.

        Expects text with sections marked by **Worked**, **Failed**, **Why**, etc.

        Args:
            iteration: Iteration number
            reflection_text: LLM-generated reflection text

        Returns:
            Parsed Reflection object
        """
        worked = []
        failed = []
        why = ""
        suggestions = []
        avoid = []

        current_section = None

        for line in reflection_text.split('\n'):
            line = line.strip()

            if line.startswith('**Worked**'):
                current_section = 'worked'
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                if content:
                    worked.append(content)
            elif line.startswith('**Failed**'):
                current_section = 'failed'
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                if content:
                    failed.append(content)
            elif line.startswith('**Why**'):
                current_section = 'why'
                why = line.split(':', 1)[1].strip() if ':' in line else ""
            elif line.startswith('**Try next**') or line.startswith('**Try Next**'):
                current_section = 'suggestions'
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                if content:
                    suggestions.append(content)
            elif line.startswith('**Avoid**'):
                current_section = 'avoid'
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                if content:
                    avoid.append(content)
            elif line.startswith(('-', '•', '*')) and current_section:
                # Bullet point - add to current section
                content = line.lstrip('-•*').strip()
                if not content:
                    continue

                if current_section == 'worked':
                    worked.append(content)
                elif current_section == 'failed':
                    failed.append(content)
                elif current_section == 'suggestions':
                    suggestions.append(content)
                elif current_section == 'avoid':
                    avoid.append(content)

        return cls(
            iteration=iteration,
            what_worked=worked,
            what_failed=failed,
            why_failed=why,
            suggestions=suggestions,
            avoid=avoid
        )


class ReflectionMemory:
    """
    Store and query reflections from past iterations.

    Implements memory component from Reflexion (Shinn et al.)

    Provides high-level learnings:
    - What approaches work
    - What to avoid
    - Suggestions for next steps
    """

    def __init__(self):
        self.reflections: List[Reflection] = []

    def add_reflection(self, reflection: Reflection):
        """Add a reflection to memory."""
        self.reflections.append(reflection)

    def get_relevant_context(self, num_recent: int = 3) -> str:
        """
        Get aggregated learnings from recent reflections.

        Use this in _team_planning_meeting() to show team what's been learned.

        Args:
            num_recent: How many recent reflections to consider

        Returns:
            Formatted string with aggregated learnings
        """
        if not self.reflections:
            return ""

        # Get recent reflections
        recent = self.reflections[-num_recent:]

        context = "## Learnings from Past Reflections\n\n"

        # Aggregate what's worked
        all_worked = []
        for r in recent:
            all_worked.extend(r.what_worked)

        if all_worked:
            context += "### What Has Worked:\n"
            # Show most recent 5
            for item in all_worked[-5:]:
                context += f"- {item}\n"

        # Aggregate what to avoid
        all_avoid = []
        for r in recent:
            all_avoid.extend(r.avoid)

        if all_avoid:
            context += "\n### What to Avoid:\n"
            for item in all_avoid[-5:]:
                context += f"- {item}\n"

        # Recent suggestions
        all_suggestions = []
        for r in recent:
            all_suggestions.extend(r.suggestions)

        if all_suggestions:
            context += "\n### Suggestions to Try:\n"
            for item in all_suggestions[-5:]:
                context += f"- {item}\n"

        return context

    def query_similar_failures(self, current_error: str) -> str:
        """
        Find past reflections about similar failures.

        Use this in _fix_code_error() to provide relevant context.

        Args:
            current_error: Current error message to match against

        Returns:
            Formatted string with similar past failures, or empty if none found
        """
        relevant = []

        # Simple keyword matching
        current_error_lower = current_error.lower()

        for r in self.reflections:
            # Check if any failure description overlaps with current error
            for fail in r.what_failed:
                if fail.lower() in current_error_lower or current_error_lower in fail.lower():
                    relevant.append(r)
                    break

        if not relevant:
            return ""

        context = "## Past Reflections on Similar Failures\n\n"

        # Show most recent 3
        for r in relevant[-3:]:
            context += f"**Iteration {r.iteration}:**\n"
            context += f"- Failed: {', '.join(r.what_failed)}\n"
            if r.why_failed:
                context += f"- Why: {r.why_failed}\n"
            if r.suggestions:
                context += f"- Suggested: {', '.join(r.suggestions)}\n"
            context += "\n"

        return context

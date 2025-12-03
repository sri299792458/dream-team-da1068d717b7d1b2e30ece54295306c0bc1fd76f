"""
Context management for robust iteration tracking.

Implements Reflexion-based learning (Shinn et al.):
- After each iteration, explicitly reflect on what happened
- Store structured learnings in memory
- Use past reflections to guide future iterations
"""

from dataclasses import dataclass, field
from typing import List, Optional





@dataclass
class Reflection:
    """
    Reflection on an iteration - scientific analysis of what was learned.

    After each iteration, the PI reflects on what happened and captures
    learnings to guide the team's future decisions.

    Workflow:
        execute → metrics → REFLECT → store learnings → team uses in planning
    """
    iteration: int
    raw_text: str  # Full reflection from PI
    key_insights: List[str] = field(default_factory=list)  # Extracted insights
    dead_ends: List[str] = field(default_factory=list)  # What to avoid

    @classmethod  
    def from_text(cls, iteration: int, reflection_text: str) -> "Reflection":
        """
        Parse reflection text and extract key insights.

        Args:
            iteration: Iteration number
            reflection_text: LLM-generated reflection text

        Returns:
            Reflection object with raw text and extracted insights
        """
        # Store raw text
        raw = reflection_text.strip()
        
        # Extract key insights (simple heuristic: look for patterns)
        insights = []
        dead_ends = []
        
        current_section = None
        lines = raw.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            line_lower = line.lower()
            if 'understanding' in line_lower or 'teach us' in line_lower:
                current_section = 'insights'
            elif 'attribution' in line_lower or 'drove' in line_lower:
                current_section = 'insights'
            elif 'constraints' in line_lower or 'limitations' in line_lower:
                current_section = 'insights'
            elif 'dead end' in line_lower or 'rule out' in line_lower:
                current_section = 'dead_ends'
            
            # Extract bullet points
            if line.startswith(('-', '•', '*', '→')):
                content = line.lstrip('-•*→').strip()
                if content:
                    if current_section == 'dead_ends':
                        dead_ends.append(content)
                    elif current_section == 'insights':
                        insights.append(content)
            # Also capture numbered lists
            elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
                content = line[2:].strip()
                if content and current_section:
                    if current_section == 'dead_ends':
                        dead_ends.append(content)
                    elif current_section == 'insights':
                        insights.append(content)
        
        return cls(
            iteration=iteration,
            raw_text=raw,
            key_insights=insights if insights else ["See raw reflection text"],
            dead_ends=dead_ends
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

        context = "## Recent Experiment Reflections\n\n"

        # Show each recent reflection
        for r in recent:
            context += f"### Iteration {r.iteration}\n"
            
            # Show key insights if extracted
            if r.key_insights and r.key_insights != ["See raw reflection text"]:
                context += "**Key Insights:**\n"
                for insight in r.key_insights[:3]:  # Top 3
                    context += f"- {insight}\n"
                context += "\n"
            
            # Show dead ends
            if r.dead_ends:
                context += "**Dead Ends:**\n"
                for dead_end in r.dead_ends[:3]:  # Top 3
                    context += f"- {dead_end}\n"
                context += "\n"
            
            # If no structured insights extracted, show snippet of raw text
            if not r.key_insights or r.key_insights == ["See raw reflection text"]:
                # Show first 200 chars of reflection
                snippet = r.raw_text[:200] + "..." if len(r.raw_text) > 200 else r.raw_text
                context += f"_{snippet}_\n\n"

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

        # Simple keyword matching against raw reflection text
        current_error_lower = current_error.lower()

        for r in self.reflections:
            # Check if raw text mentions similar error or dead ends mention it
            if current_error_lower in r.raw_text.lower():
                relevant.append(r)
            elif any(current_error_lower in de.lower() for de in r.dead_ends):
                relevant.append(r)

        if not relevant:
            return ""

        context = "## Past Reflections on Similar Issues\n\n"

        # Show most recent 3
        for r in relevant[-3:]:
            context += f"**Iteration {r.iteration}:**\n"
            
            # Show relevant dead ends
            if r.dead_ends:
                context += f"Dead ends identified: {', '.join(r.dead_ends[:2])}\n"
            
            # Show snippet of raw reflection
            snippet = r.raw_text[:200] + "..." if len(r.raw_text) > 200 else r.raw_text
            context += f"_{snippet}_\n\n"

        return context

"""
Dream Team: A dynamic multi-agent framework with evolving personas.

This framework enables AI agents to:
- Evolve their expertise based on problem needs
- Integrate research from Semantic Scholar
- Collaborate through structured meetings
- Build knowledge bases that grow over time
- Use mathematical state for emergent evolution
"""

from .agent import Agent, Paper, KnowledgeBase
from .llm import GeminiLLM, get_llm
from .research import SemanticScholarAPI, ResearchAssistant, get_research_assistant
from .evolution_agent import EvolutionAgent, EvolutionDecision
from .meetings import TeamMeeting, IndividualMeeting
from .executor import CodeExecutor, extract_code_from_text
from .orchestrator import ExperimentOrchestrator
from .utils import save_json, load_json, load_summaries
from .serialization import RobustJSONEncoder, robust_dump, robust_dumps

# 5-Layer Architecture Components
from .event_store import EventStore, Event
from .semantic_state import IterationRecord
from .context import ExecutionContext, ExperimentState
from .nodes import (
    create_supervisor_node,
    create_coding_node,
    create_execution_node,
    create_reflection_node,
    create_evolution_node,
    create_meeting_node
)


__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Paper",
    "KnowledgeBase",
    "GeminiLLM",
    "get_llm",
    "SemanticScholarAPI",
    "ResearchAssistant",
    "get_research_assistant",
    "EvolutionAgent",
    "EvolutionDecision",
    "TeamMeeting",
    "IndividualMeeting",
    "CodeExecutor",
    "extract_code_from_text",
    "ExperimentOrchestrator",
    "save_json",
    "load_json",
    "load_summaries",
    "RobustJSONEncoder",
    "robust_dump",
    "robust_dumps",
    # 5-Layer Architecture
    "EventStore",
    "Event",
    "IterationRecord",
    "ContextBuilder",
]

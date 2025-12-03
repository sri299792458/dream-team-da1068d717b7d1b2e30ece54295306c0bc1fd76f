"""
Agent module for Dream Team framework.

Implements evolving agents with growing knowledge bases and mathematical state.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json
import numpy as np


@dataclass
class Paper:
    """Research paper representation"""
    title: str
    authors: List[str]
    year: int
    abstract: str
    key_findings: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    semantic_scholar_id: Optional[str] = None
    citation_count: int = 0
    applied: bool = False
    impact_notes: Optional[str] = None

    def to_dict(self):
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "key_findings": self.key_findings,
            "techniques": self.techniques,
            "relevance": self.relevance_score,
            "citation_count": self.citation_count,
            "applied": self.applied,
            "impact": self.impact_notes
        }


@dataclass
class KnowledgeBase:
    """Agent's accumulated knowledge"""
    domain_facts: List[str] = field(default_factory=list)
    papers: List[Paper] = field(default_factory=list)
    techniques_mastered: List[str] = field(default_factory=list)
    error_insights: List[str] = field(default_factory=list)
    successful_patterns: List[str] = field(default_factory=list)

    def add_paper(self, paper: Paper):
        """Add paper, avoiding duplicates"""
        if not any(p.title == paper.title for p in self.papers):
            self.papers.append(paper)

    def add_fact(self, fact: str, source: Optional[str] = None):
        """Add domain fact with optional citation"""
        fact_with_source = f"{fact} [Source: {source}]" if source else fact
        if fact_with_source not in self.domain_facts:
            self.domain_facts.append(fact_with_source)

    def add_technique(self, technique: str):
        if technique not in self.techniques_mastered:
            self.techniques_mastered.append(technique)
    
    def add_success_pattern(self, iteration: int, technique: str, metric: str, improvement: float):
        """Track successful pattern."""
        pattern = f"Iter {iteration}: {technique} improved {metric} by {improvement:.4f}"
        if pattern not in self.successful_patterns:
            self.successful_patterns.append(pattern)
    
    def add_failure_pattern(self, iteration: int, technique: str, metric: str, reason: str):
        """Track failure pattern."""
        pattern = f"Iter {iteration}: {technique} failed on {metric} - {reason}"
        if pattern not in self.error_insights:
            self.error_insights.append(pattern)
    
    def add_error_insight(self, iteration: int, error_type: str, error_msg: str, solution: Optional[str] = None):
        """Add error catalog entry."""
        insight = f"Iter {iteration}: {error_type} - {error_msg}"
        if solution:
            insight += f" [Solution: {solution}]"
        if insight not in self.error_insights:
            self.error_insights.append(insight)
    
    def collect_for_intent(self, intent: str, max_items: int = 5) -> Dict[str, List[str]]:
        """
        Retrieve relevant knowledge for specific purpose.
        
        Args:
            intent: Purpose of retrieval ("plan_next_iteration", "fix_error", "code_implementation")
            max_items: Maximum items per category
        
        Returns:
            Dictionary with categorized knowledge
        """
        result = {
            "techniques": [],
            "patterns": [],
            "pitfalls": []
        }
        
        if intent == "plan_next_iteration":
            # Most recent successful patterns
            result["patterns"] = self.successful_patterns[-max_items:]
            # Known pitfalls to avoid
            result["pitfalls"] = self.error_insights[-max_items:]
            # Techniques mastered
            result["techniques"] = self.techniques_mastered[-max_items:]
        
        elif intent == "code_implementation":
            # Successful patterns for guidance
            result["patterns"] = self.successful_patterns[-max_items:]
            # Techniques to use
            result["techniques"] = self.techniques_mastered[-max_items:]
        
        elif intent == "fix_error":
            # Error insights and solutions
            result["pitfalls"] = self.error_insights[-max_items:]
            # Successful patterns that worked
            result["patterns"] = self.successful_patterns[-max_items:]
        
        return result

    def to_dict(self):
        return {
            "domain_facts": self.domain_facts,
            "papers": [p.to_dict() for p in self.papers],
            "techniques": self.techniques_mastered,
            "error_insights": self.error_insights,
            "successful_patterns": self.successful_patterns
        }


@dataclass
class AgentSnapshot:
    """Snapshot of agent state at a point in time"""
    timestamp: str
    title: str
    expertise: str
    role: str
    specialization_depth: int
    knowledge_base_summary: Dict
    trigger_reason: Optional[str] = None
    # Mathematical state
    gini_coefficient: float = 0.0
    max_depth: float = 0.0
    top_concepts: List[Tuple[str, float]] = field(default_factory=list)


class Agent:
    """Evolving agent with mathematical state"""

    def __init__(
        self,
        title: str,
        expertise: str,
        goal: str,
        role: str,
        model: str = "gemini-2.5-flash",
        specialization_depth: int = 0
    ):
        # Traditional attributes
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model
        self.specialization_depth = specialization_depth

        self.knowledge_base = KnowledgeBase()
        self.evolution_history: List[AgentSnapshot] = []

        # Track contributions
        self.meetings_participated = 0
        self.experiments_proposed = 0
        self.successful_contributions = 0



    @property
    def prompt(self) -> str:
        """
        Generate system prompt WITHOUT knowledge base injection.
        
        Knowledge base is now provided separately by ContextBuilder.
        This keeps the base prompt stable and context dynamic.
        """
        # Role-specific closing to activate relevant expertise
        if "Principal Investigator" in self.title or "Lead" in self.title:
            closing = """You are part of a research team solving data science challenges.

As the lead, ensure your approach embodies rigorous experimental methodology. Consider what constitutes sound scientific practice: how to validate hypotheses, establish baselines, make incremental progress, and learn systematically from results. Apply your full expertise to guide the team toward methodologically sound decisions."""
        else:
            closing = """You are part of a research team solving data science challenges.

When proposing approaches, consider both domain expertise and experimental rigor. Think about validation, incremental progress, and learning from previous work. Provide insightful, actionable contributions grounded in sound methodology."""

        base_prompt = f"""You are {self.title}.

Expertise: {self.expertise}

Goal: {self.goal}

Role: {self.role}

{closing}"""

        return base_prompt

    def snapshot(self, trigger_reason: Optional[str] = None) -> AgentSnapshot:
        """Take snapshot of current state"""
        return AgentSnapshot(
            timestamp=datetime.now().isoformat(),
            title=self.title,
            expertise=self.expertise,
            role=self.role,
            specialization_depth=self.specialization_depth,
            knowledge_base_summary=self.knowledge_base.to_dict(),
            trigger_reason=trigger_reason,
            gini_coefficient=0.0,
            max_depth=0.0,
            top_concepts=[]
        )

    def save(self, filepath: str):
        """Save agent state to JSON"""
        state = {
            "title": self.title,
            "expertise": self.expertise,
            "goal": self.goal,
            "role": self.role,
            "model": self.model,
            "specialization_depth": self.specialization_depth,
            "knowledge_base": self.knowledge_base.to_dict(),
            "evolution_history": [
                {
                    "timestamp": s.timestamp,
                    "title": s.title,
                    "expertise": s.expertise,
                    "role": s.role,
                    "depth": s.specialization_depth,
                    "trigger": s.trigger_reason,
                    "gini": s.gini_coefficient,
                    "max_depth": s.max_depth
                }
                for s in self.evolution_history
            ],
            "stats": {
                "meetings_participated": self.meetings_participated,
                "experiments_proposed": self.experiments_proposed,
                "successful_contributions": self.successful_contributions
            },

        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Agent':
        """Load agent from JSON"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        agent = cls(
            title=state["title"],
            expertise=state["expertise"],
            goal=state["goal"],
            role=state["role"],
            model=state.get("model", "gemini-2.5-flash"),
            specialization_depth=state.get("specialization_depth", 0)
        )

        # Restore knowledge base
        kb_data = state.get("knowledge_base", {})
        agent.knowledge_base.domain_facts = kb_data.get("domain_facts", [])
        agent.knowledge_base.techniques_mastered = kb_data.get("techniques", [])
        agent.knowledge_base.error_insights = kb_data.get("error_insights", [])
        agent.knowledge_base.successful_patterns = kb_data.get("successful_patterns", [])

        for p_data in kb_data.get("papers", []):
            paper = Paper(
                title=p_data["title"],
                authors=p_data.get("authors", []),
                year=p_data.get("year", 2024),
                abstract="",
                key_findings=p_data.get("key_findings", []),
                techniques=p_data.get("techniques", []),
                relevance_score=p_data.get("relevance", 0.0),
                applied=p_data.get("applied", False),
                impact_notes=p_data.get("impact")
            )
            agent.knowledge_base.papers.append(paper)



        # Restore stats
        stats = state.get("stats", {})
        agent.meetings_participated = stats.get("meetings_participated", 0)
        agent.experiments_proposed = stats.get("experiments_proposed", 0)
        agent.successful_contributions = stats.get("successful_contributions", 0)

        return agent

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary (for event storage)"""
        return {
            "title": self.title,
            "expertise": self.expertise,
            "goal": self.goal,
            "role": self.role,
            "model": self.model,
            "specialization_depth": self.specialization_depth,
            "knowledge_base": self.knowledge_base.to_dict(),
            "evolution_history": [
                {
                    "timestamp": s.timestamp,
                    "title": s.title,
                    "expertise": s.expertise,
                    "role": s.role,
                    "depth": s.specialization_depth,
                    "trigger": s.trigger_reason,
                    "gini": s.gini_coefficient,
                    "max_depth": s.max_depth
                }
                for s in self.evolution_history
            ],
            "stats": {
                "meetings_participated": self.meetings_participated,
                "experiments_proposed": self.experiments_proposed,
                "successful_contributions": self.successful_contributions
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Restore agent from dictionary (for event restoration)"""
        agent = cls(
            title=data["title"],
            expertise=data["expertise"],
            goal=data["goal"],
            role=data["role"],
            model=data.get("model", "gemini-2.5-flash"),
            specialization_depth=data.get("specialization_depth", 0)
        )

        # Restore knowledge base
        kb_data = data.get("knowledge_base", {})
        agent.knowledge_base.domain_facts = kb_data.get("domain_facts", [])
        agent.knowledge_base.techniques_mastered = kb_data.get("techniques", [])
        agent.knowledge_base.error_insights = kb_data.get("error_insights", [])
        agent.knowledge_base.successful_patterns = kb_data.get("successful_patterns", [])

        for p_data in kb_data.get("papers", []):
            paper = Paper(
                title=p_data["title"],
                authors=p_data.get("authors", []),
                year=p_data.get("year", 2024),
                abstract="",
                key_findings=p_data.get("key_findings", []),
                techniques=p_data.get("techniques", []),
                relevance_score=p_data.get("relevance", 0.0),
                applied=p_data.get("applied", False),
                impact_notes=p_data.get("impact")
            )
            agent.knowledge_base.papers.append(paper)

        # Restore stats
        stats = data.get("stats", {})
        agent.meetings_participated = stats.get("meetings_participated", 0)
        agent.experiments_proposed = stats.get("experiments_proposed", 0)
        agent.successful_contributions = stats.get("successful_contributions", 0)

        return agent




    # Knowledge integration

    def add_paper_to_knowledge(self, paper: Paper):
        """Integrate paper into knowledge base"""
        self.knowledge_base.add_paper(paper)

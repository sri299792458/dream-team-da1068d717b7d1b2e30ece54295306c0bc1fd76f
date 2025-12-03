"""
Context builder for Dream Team framework.

Centralizes all context construction for different agent interactions.
Replaces scattered context building and truncation logic.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from .semantic_state import IterationRecord
from .agent import Agent, KnowledgeBase
from .context import ReflectionMemory


class ContextBuilder:
    """
    Centralized context management.
    
    Single source of truth for what context flows to agents in different scenarios.
    Prevents scattered truncation/summarization logic.
    """
    
    def __init__(
        self,
        experiment_id: str,
        reflection_memory: Optional[ReflectionMemory] = None,
        executor=None,  # CodeExecutor (avoid circular import)
        artifacts_dir: Optional[Path] = None
    ):
        """
        Initialize context builder.
        
        Args:
            experiment_id: Experiment identifier
            reflection_memory: Memory of past reflections
            executor: CodeExecutor instance for execution state access
            artifacts_dir: Directory where artifacts are saved
        """
        self.experiment_id = experiment_id
        self.reflection_memory = reflection_memory or ReflectionMemory()
        self.executor = executor
        self.artifacts_dir = artifacts_dir
        
        # Will be populated during orchestration
        self.iterations: List[IterationRecord] = []
        self.evolution_gaps: List[str] = []
        self.evolution_coverage: Dict[str, float] = {}
    
    def set_iterations(self, iterations: List[IterationRecord]):
        """Update iteration records."""
        self.iterations = iterations
    
    def set_evolution_state(self, gaps: List[str], coverage: Dict[str, float]):
        """Update evolution gaps and coverage."""
        self.evolution_gaps = gaps
        self.evolution_coverage = coverage
    
    def for_team_meeting(
        self,
        target_metric: str,
        num_recent_iterations: int = 3
    ) -> str:
        """
        Build context for team planning meeting.
        
        Includes:
        - Recent iteration summaries
        - KB learnings
        - Evolution gaps/coverage
        - Past reflections
        - Current Data Schemas (NEW)
        - Saved Artifacts (NEW)
        
        Args:
            target_metric: Metric being optimized
            num_recent_iterations: Number of recent iterations to include
        
        Returns:
            Formatted context string
        """
        context = "# Context for Team Planning\n\n"
        
        # NEW: Current Data Schemas (Critical for preventing hallucinations)
        current_schemas = self._build_column_schemas()
        if current_schemas:
            context += "## Current Data Schemas\n\n"
            context += current_schemas
            
        # NEW: Saved Artifacts (Useful for planning reuse)
        artifacts = self._build_artifact_inventory()
        if artifacts:
            context += artifacts

        # Recent iterations
        if self.iterations:
            context += "## Recent Iterations\n\n"
            recent = self.iterations[-num_recent_iterations:]
            for it in recent:
                context += f"### Iteration {it.iteration}\n"
                context += f"**Approach:** {it.approach}\n\n"
                
                # Metrics
                if target_metric in it.metrics:
                    context += f"**{target_metric}:** {it.metrics[target_metric]:.4f}\n"
                
                # Key observations from analysis
                if it.output_analysis.key_observations:
                    context += "**Observations:**\n"
                    for obs in it.output_analysis.key_observations[:3]:
                        context += f"- {obs}\n"
                
                # Techniques used
                if it.code_analysis and it.code_analysis.techniques:
                    context += f"**Techniques:** {', '.join(it.code_analysis.techniques[:5])}\n"
                
                context += "\n"
        
        # Evolution information
        if self.evolution_gaps:
            context += "## Evolution Analysis\n\n"
            context += f"**Underserved concepts:** {', '.join(self.evolution_gaps[:5])}\n"
            context += "Consider techniques targeting these areas.\n\n"
        
        if self.evolution_coverage:
            context += "**Strong coverage:** "
            top_covered = sorted(self.evolution_coverage.items(), key=lambda x: x[1], reverse=True)[:3]
            context += ', '.join([f"{k} ({v:.2f})" for k, v in top_covered])
            context += "\n\n"
        
        # Reflection learnings
        reflection_context = self.reflection_memory.get_relevant_context(num_recent=num_recent_iterations)
        if reflection_context:
            context += reflection_context + "\n"
        
        return context
    
    def for_coding(
        self,
        coding_agent: Agent,
        team_plan: str,
        target_metric: str,
        column_schemas: str = ""
    ) -> str:
        """
        Build context for code implementation.
        
        Includes:
        - Team plan
        - Execution state (variables, DataFrames, artifacts)
        - Last iteration's analyses
        - KB successful patterns
        
        Args:
            coding_agent: The agent who will write code
            team_plan: Plan from team meeting
            target_metric: Metric being optimized
            column_schemas: DEPRECATED - now auto-extracted from execution context
        
        Returns:
            Formatted context string
        """
        context = "# Context for Implementation\n\n"
        
        # NEW: Execution State
        var_inventory = self._build_variable_inventory()
        if var_inventory:
            context += var_inventory
        
        # NEW: Current DataFrame schemas (auto-extracted)
        current_schemas = self._build_column_schemas()
        if current_schemas:
            context += "## Current Data Schemas\n\n"
            context += current_schemas
        elif column_schemas:  # Fallback to passed schemas (bootstrap)
            context += "## Data Schema (CRITICAL: Use EXACT column names)\n\n"
            context += column_schemas + "\n\n"
        
        # NEW: Saved artifacts
        artifacts = self._build_artifact_inventory()
        if artifacts:
            context += artifacts
        
        # Team plan
        context += "## Team Plan\n\n"
        context += team_plan + "\n\n"
        
        # Last iteration insights
        if self.iterations:
            last = self.iterations[-1]
            context += "## Last Iteration Insights\n\n"
            
            # What worked
            if last.output_analysis.key_observations:
                context += "**Key Observations:**\n"
                for obs in last.output_analysis.key_observations:
                    context += f"- {obs}\n"
                context += "\n"
            
            # Errors to avoid
            if last.output_analysis.errors:
                context += "**Errors Encountered:**\n"
                for err in last.output_analysis.errors[:3]:
                    context += f"- {err}\n"
                context += "\n"
        
        # Knowledge base patterns
        kb_context = coding_agent.knowledge_base.collect_for_intent("code_implementation", max_items=5)
        if kb_context.get("techniques"):
            context += "## Proven Techniques\n\n"
            for tech in kb_context["techniques"]:
                context += f"- {tech}\n"
            context += "\n"
        
        if kb_context.get("patterns"):
            context += "## Successful Patterns\n\n"
            for pattern in kb_context["patterns"]:
                context += f"- {pattern}\n"
            context += "\n"
        
        return context
    
    def for_error_fix(
        self,
        coding_agent: Agent,
        failed_code: str,
        error: str,
        traceback: str,
        approach: str
    ) -> str:
        """
        Build context for error fixing.
        
        Includes:
        - Error details
        - Failed code (NEW)
        - Execution state (NEW)
        - Similar past errors
        - Last output analysis
        - Relevant techniques
        
        Args:
            coding_agent: Agent who will fix the error
            failed_code: Code that failed
            error: Error message
            traceback: Stack trace
            approach: Original approach description
        
        Returns:
            Formatted context string
        """
        context = "# Context for Error Fix\n\n"
        
        # Error information
        context += "## Error Details\n\n"
        context += f"**Error:** {error}\n\n"
        context += "**Traceback:**\n```\n"
        context += traceback[:1000] + ("..." if len(traceback) > 1000 else "")
        context += "\n```\n\n"
        
        # NEW: Failed Code (Critical for debugging)
        context += "## Failed Code\n```python\n"
        context += failed_code
        context += "\n```\n\n"
        
        # NEW: Execution State (Critical for knowing what exists)
        var_inventory = self._build_variable_inventory()
        if var_inventory:
            context += var_inventory
            
        current_schemas = self._build_column_schemas()
        if current_schemas:
            context += "## Current Data Schemas\n\n"
            context += current_schemas
            
        artifacts = self._build_artifact_inventory()
        if artifacts:
            context += artifacts
        
        # Original approach
        context += "## Original Approach\n\n"
        context += approach + "\n\n"
        
        # Similar past errors
        similar_errors = self.reflection_memory.query_similar_failures(error)
        if similar_errors:
            context += similar_errors + "\n"
        
        # Last iteration analysis
        if self.iterations:
            last = self.iterations[-1]
            if last.output_analysis.errors:
                context += "## Recent Error Patterns\n\n"
                for err in last.output_analysis.errors[:3]:
                    context += f"- {err}\n"
                context += "\n"
        
        # KB error insights
        kb_context = coding_agent.knowledge_base.collect_for_intent("fix_error", max_items=5)
        if kb_context.get("pitfalls"):
            context += "## Known Pitfalls to Avoid\n\n"
            for pitfall in kb_context["pitfalls"]:
                context += f"- {pitfall}\n"
            context += "\n"
        
        return context
    
    def for_reflection(
        self,
        iter_record: IterationRecord,
        target_metric: str
    ) -> str:
        """
        Build context for reflection.
        
        Reflection sees everything from this iteration (no truncation).
        
        Args:
            iter_record: Complete iteration record
            target_metric: Metric being optimized
        
        Returns:
            Formatted context string
        """
        context = "# Context for Reflection\n\n"
        
        context += f"## Iteration {iter_record.iteration}\n\n"
        
        # Approach
        context += "## Approach\n\n"
        context += iter_record.approach + "\n\n"
        
        # Code Analysis
        if iter_record.code_analysis:
            context += "## Code Analysis\n\n"
            context += f"**Complexity:** {iter_record.code_analysis.complexity}\n"
            
            if iter_record.code_analysis.techniques:
                context += f"**Techniques used:** {', '.join(iter_record.code_analysis.techniques)}\n"
            
            if iter_record.code_analysis.key_decisions:
                context += "**Key decisions:**\n"
                for decision in iter_record.code_analysis.key_decisions:
                    context += f"- {decision}\n"
            
            context += "\n"
        
        # Output Analysis
        context += "## Execution Analysis\n\n"
        context += f"**Summary:** {iter_record.output_analysis.raw_summary}\n\n"
        
        if iter_record.output_analysis.key_observations:
            context += "**Observations:**\n"
            for obs in iter_record.output_analysis.key_observations:
                context += f"- {obs}\n"
            context += "\n"
        
        if iter_record.output_analysis.errors:
            context += "**Errors:**\n"
            for err in iter_record.output_analysis.errors:
                context += f"- {err}\n"
            context += "\n"
        
        if iter_record.output_analysis.warnings:
            context += "**Warnings:**\n"
            for warn in iter_record.output_analysis.warnings:
                context += f"- {warn}\n"
            context += "\n"
        
        # Metrics
        context += "## Metrics\n\n"
        if target_metric in iter_record.metrics:
            context += f"**{target_metric}:** {iter_record.metrics[target_metric]:.4f}\n"
        
        for key, value in iter_record.metrics.items():
            if key != target_metric:
                context += f"**{key}:** {value:.4f}\n"
        
        return context
    
    # ===== Execution State Helpers =====
    
    def _build_variable_inventory(self) -> str:
        """Build inventory of available variables from executor."""
        if not self.executor or not self.executor.data_context:
            return ""
        
        context = "## Available Variables\n\n"
        context += "Variables currently in execution context:\n\n"
        
        import pandas as pd
        import numpy as np
        
        for name, value in self.executor.data_context.items():
            # Skip internal vars
            if name.startswith('_') or name in ['pd', 'np', 'torch', 'Path', 'artifacts_dir']:
                continue
            
            # Describe the type and shape
            if isinstance(value, pd.DataFrame):
                context += f"- `{name}`: DataFrame ({value.shape[0]} rows, {value.shape[1]} cols)\n"
            elif isinstance(value, pd.Series):
                context += f"- `{name}`: Series ({len(value)} values)\n"
            elif isinstance(value, np.ndarray):
                context += f"- `{name}`: ndarray {value.shape}\n"
            elif hasattr(value, '__class__'):
                class_name = value.__class__.__name__
                # Check if it's a fitted model
                if hasattr(value, 'predict') or hasattr(value, 'fit'):
                    context += f"- `{name}`: {class_name} (fitted model)\n"
                else:
                    context += f"- `{name}`: {class_name}\n"
            else:
                context += f"- `{name}`: {type(value).__name__}\n"
        
        context += "\n"
        return context
    
    def _build_column_schemas(self) -> str:
        """Extract current DataFrame column schemas from execution context."""
        if not self.executor or not self.executor.data_context:
            return ""
        
        import pandas as pd
        context = ""
        
        # Find DataFrames in context
        for name, value in self.executor.data_context.items():
            if isinstance(value, pd.DataFrame):
                context += f"### {name} Columns\n\n"
                for col in value.columns:
                    dtype = str(value[col].dtype)
                    context += f"- `{col}`: {dtype}\n"
                context += "\n"
        
        return context
    
    def _build_artifact_inventory(self) -> str:
        """Build inventory of saved artifacts."""
        if not self.artifacts_dir or not self.artifacts_dir.exists():
            return ""
        
        # Scan for common artifact types
        artifacts = list(self.artifacts_dir.glob("*.pkl")) + \
                   list(self.artifacts_dir.glob("*.joblib")) + \
                   list(self.artifacts_dir.glob("*.pt")) + \
                   list(self.artifacts_dir.glob("*.pth"))
        
        if not artifacts:
            return ""
        
        context = "## Saved Artifacts\n\n"
        context += "Models and objects saved from previous iterations:\n\n"
        
        for artifact in artifacts:
            size_kb = artifact.stat().st_size / 1024
            context += f"- `{artifact.name}` ({size_kb:.1f} KB)\n"
        
        context += "\nLoad with: `joblib.load(artifacts_dir / 'filename.pkl')` or `torch.load(artifacts_dir / 'filename.pt')`\n\n"
        
        return context

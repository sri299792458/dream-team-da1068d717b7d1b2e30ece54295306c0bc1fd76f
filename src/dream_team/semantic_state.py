"""
Semantic state models for Dream Team framework.

Defines semantic understanding of iterations: output analysis, code analysis,
and complete iteration records.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class IterationRecord:
    """
    Complete state of one iteration.
    
    This is the canonical "state of experiment" that replaces scattered JSON.
    Contains both raw data and semantic analyses.
    """
    iteration: int
    approach: str
    code: str
    results: Dict[str, Any]
    metrics: Dict[str, float]
    
    # Semantic analyses
    reflection: str  # Or parsed Reflection object
    
    # Optional metadata
    timestamp: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        """Load from dictionary."""
        # Handle legacy data with output_analysis
        if "output_analysis" in data:
            data.pop("output_analysis")
        
        # Remove code_analysis if present in legacy data
        if "code_analysis" in data:
            data.pop("code_analysis")
        
        return cls(**data)
    
    def get_metric(self, metric_name: str) -> Optional[float]:
        """
        Get metric value by name.
        
        Args:
            metric_name: Name of metric to retrieve
        
        Returns:
            Metric value or None if not found
        """
        return self.metrics.get(metric_name)
    
    def is_improvement(self, other: "IterationRecord", metric_name: str, minimize: bool = True) -> bool:
        """
        Check if this iteration improved over another.
        
        Args:
            other: Other iteration to compare against
            metric_name: Metric to compare
            minimize: True if lower is better, False if higher is better
        
        Returns:
            True if this iteration is better
        """
        this_val = self.get_metric(metric_name)
        other_val = other.get_metric(metric_name)
        
        if this_val is None or other_val is None:
            return False
        
        if minimize:
            return this_val < other_val
        else:
            return this_val > other_val

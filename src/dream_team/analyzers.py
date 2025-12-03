"""
Analyzers for Dream Team framework.

LLM-based semantic analysis of code and execution outputs.
Replaces simple truncation with intelligent understanding.
"""

from typing import Dict, Any, Optional
from .semantic_state import OutputAnalysis
from .llm import GeminiLLM
import json
import re


class OutputAnalyzer:
    """
    Analyze execution output using LLM for semantic understanding.
    
    Extracts:
    - Errors and warnings
    - Metrics
    - Key observations (overfitting, data leakage, etc.)
    - Stack traces
    """
    
    def __init__(self, llm: Optional[GeminiLLM] = None):
        """
        Initialize analyzer.
        
        Args:
            llm: LLM instance for analysis (creates default if None)
        """
        self.llm = llm or GeminiLLM(model_name="gemini-3-pro-preview")
    
    def analyze(
        self,
        output: str,
        error: Optional[str] = None,
        traceback: Optional[str] = None
    ) -> OutputAnalysis:
        """
        Analyze execution output.
        
        Args:
            output: Full execution output (stdout)
            error: Error message if execution failed
            traceback: Stack trace if execution failed
        
        Returns:
            OutputAnalysis with semantic understanding
        """
        # Build analysis prompt
        prompt = self._build_analysis_prompt(output, error, traceback)
        
        # Request structured output
        try:
            data = self.llm.generate_json(
                prompt,
                system_instruction="You are an expert code execution analyzer. Extract semantic information from execution outputs."
            )
        except Exception:
            # Fallback to basic analysis
            return self._fallback_analysis(output, error, traceback)
        
        # Build OutputAnalysis
        return OutputAnalysis(
            raw_summary=data.get("summary", "Execution completed"),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            metrics_found=data.get("metrics", {}),
            key_observations=data.get("observations", []),
            stack_traces=[traceback] if traceback else []
        )
    
    def _build_analysis_prompt(
        self,
        output: str,
        error: Optional[str],
        traceback: Optional[str]
    ) -> str:
        """Build prompt for LLM analysis."""
        prompt = f"""Analyze code execution results.

Task: {{task_description}}

Code:
{{code}}

Output:
{{output}}

Error (if any):
{{error}}
{{traceback}}

Provide analysis as valid JSON only, no other text:
{{{{
    "success": true or false,
    "summary": "1-2 sentence summary",
    "key_observations": ["obs1", "obs2"],
    "errors": [],
    "warnings": []
}}}}

Rules:
- All fields required. Use empty arrays [] if none.
- No markdown, no ```json blocks.
""".format(
            task_description="Code Execution", # Context not passed in this method, generic label
            code="[Code not included in prompt to save tokens]" if len(output) > 10000 else "See above", # Placeholder as code isn't passed to this method
            output=output,
            error=error or "",
            traceback=traceback or ""
        )
        
        return prompt
    
    def _fallback_analysis(
        self,
        output: str,
        error: Optional[str],
        traceback: Optional[str]
    ) -> OutputAnalysis:
        """Fallback analysis if LLM fails."""
        errors = []
        if error:
            errors.append(error)
        
        # Extract common error patterns
        if "NameError" in output:
            errors.append("Variable not defined")
        if "KeyError" in output:
            errors.append("Missing dictionary key")
        if "FileNotFoundError" in output:
            errors.append("File not found")
        
        # Try to extract metrics from output
        metrics = {}
        metric_patterns = [
            r"(\w+)\s*[:=]\s*([\d.]+)",
            r"(\w+)\s*score\s*[:=]\s*([\d.]+)"
        ]
        for pattern in metric_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for name, value in matches:
                try:
                    metrics[name.lower()] = float(value)
                except ValueError:
                    pass
        
        summary = "Execution failed" if error else "Execution completed"
        
        return OutputAnalysis(
            raw_summary=summary,
            errors=errors,
            warnings=[],
            metrics_found=metrics,
            key_observations=[],
            stack_traces=[traceback] if traceback else []
        )

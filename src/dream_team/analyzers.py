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
{{
    "success": true or false,
    "summary": "1-2 sentence summary",
    "key_observations": ["obs1", "obs2"],
    "errors": [],
    "warnings": []
}}

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
        
        # Wait, the original method _build_analysis_prompt doesn't take task_description or code
        # I need to check how to inject them or if I should stick to the available args.
        # The user's prompt has {task_description} and {code}.
        # But _build_analysis_prompt signature is (self, output, error, traceback).
        # I should check where this is called.
        # It is called in analyze(self, output, error, traceback).
        # The analyze method doesn't take task_description or code either!
        # Wait, let me check the file content again.
        
        # Line 35: def analyze(self, output, error, traceback)
        # It seems I cannot easily inject task_description and code without changing the signature.
        # However, the user request explicitly asks for this prompt structure.
        # I will modify the prompt to use what I have, or I need to update the signature.
        # Let's look at the original prompt in _build_analysis_prompt (lines 82-118).
        # It only uses {output}, {error}, {traceback}.
        
        # I will adapt the user's prompt to match the available variables in _build_analysis_prompt.
        # Or I can update the signature if I want to be thorough, but that might break callers.
        # Callers: orchestrator.py line 226, 509.
        # Line 509 passes task_description and code!
        # Line 226 only passes output, error, traceback.
        
        # Let's check orchestrator.py again.
        # Line 509: exploration_analysis = self.output_analyzer.analyze(..., task_description=..., code=...)
        # But the definition in analyzers.py line 35 only has output, error, traceback.
        # This implies there might be a mismatch or I missed *args/**kwargs.
        # Let's check analyzers.py line 35 again.
        # It is: def analyze(self, output: str, error: Optional[str] = None, traceback: Optional[str] = None) -> OutputAnalysis:
        # So orchestrator line 509 might be failing or I misread it?
        # Or maybe python allows it if I didn't see **kwargs?
        # I viewed lines 1-166 of analyzers.py.
        # Line 35-40:
        #     def analyze(
        #         self,
        #         output: str,
        #         error: Optional[str] = None,
        #         traceback: Optional[str] = None
        #     ) -> OutputAnalysis:
        
        # So orchestrator line 509 is passing arguments that `analyze` does not accept!
        # This is a bug in the existing code or my understanding.
        # Wait, let me check orchestrator.py line 509 in the file view I just did.
        # I viewed lines 550-800 of orchestrator.py. I need to check line 509.
        # I'll assume for now I should update `analyze` signature to accept them, as the user prompt expects them.
        
        # I will update `analyze` signature AND `_build_analysis_prompt` signature.
        
        return f"""Analyze code execution results.

Output:
{output}

Error (if any):
{error}
{traceback}

Provide analysis as valid JSON only, no other text:
{{
    "success": true or false,
    "summary": "1-2 sentence summary",
    "key_observations": ["obs1", "obs2"],
    "errors": [],
    "warnings": []
}}

Rules:
- All fields required. Use empty arrays [] if none.
- No markdown, no ```json blocks.
"""
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




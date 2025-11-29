"""
Analyzers for Dream Team framework.

LLM-based semantic analysis of code and execution outputs.
Replaces simple truncation with intelligent understanding.
"""

from typing import Dict, Any, Optional
from .semantic_state import OutputAnalysis, CodeAnalysis
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
        self.llm = llm or GeminiLLM(model_name="gemini-2.5-flash")
    
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
        prompt = """Analyze this code execution output and extract semantic information.

**Output:**
```
{output}
```
""".format(output=output[:5000])  # Limit to 5k chars
        
        if error:
            prompt += f"\n**Error:** {error}\n"
        
        if traceback:
            prompt += f"\n**Traceback:**\n```\n{traceback[:2000]}\n```\n"
        
        prompt += """
Return a JSON object with:
{
  "summary": "Brief summary of what happened",
  "errors": ["error message 1", "error message 2"],
  "warnings": ["warning 1", "warning 2"],
  "metrics": {"metric_name": value},
  "observations": ["key observation 1", "key observation 2"]
}

For observations, identify issues like:
- Overfitting (train vs validation gap)
- Data leakage
- Convergence problems
- Class imbalance issues
- Memory issues
- Performance bottlenecks

Only include observations that are clearly evident in the output.
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


class CodeAnalyzer:
    """
    Analyze code structure using LLM.
    
    Extracts:
    - ML techniques used
    - Key design decisions
    - Libraries
    - Complexity assessment
    - Data flow
    """
    
    def __init__(self, llm: Optional[GeminiLLM] = None):
        """
        Initialize analyzer.
        
        Args:
            llm: LLM instance for analysis (creates default if None)
        """
        self.llm = llm or GeminiLLM(model_name="gemini-2.5-flash")
    
    def analyze(self, code: str) -> CodeAnalysis:
        """
        Analyze code structure.
        
        Args:
            code: Python code to analyze
        
        Returns:
            CodeAnalysis with semantic understanding
        """
        prompt = self._build_analysis_prompt(code)
        
        try:
            data = self.llm.generate_json(
                prompt,
                system_instruction="You are an expert code analyzer specializing in machine learning code."
            )
        except Exception:
            # Fallback to basic analysis
            return self._fallback_analysis(code)
        
        return CodeAnalysis(
            techniques=data.get("techniques", []),
            key_decisions=data.get("key_decisions", []),
            libraries_used=data.get("libraries", []),
            complexity=data.get("complexity", "moderate"),
            data_flow_summary=data.get("data_flow", "")
        )
    
    def _build_analysis_prompt(self, code: str) -> str:
        """Build prompt for code analysis."""
        return f"""Analyze this Python code and extract key information.

**Code:**
```python
{code[:8000]}
```

Return a JSON object with:
{{
  "techniques": ["technique1", "technique2"],
  "key_decisions": ["decision1", "decision2"],
  "libraries": ["library1", "library2"],
  "complexity": "simple|moderate|complex",
  "data_flow": "Brief description of data flow"
}}

For techniques, identify ML/data science techniques like:
- gradient_boosting, random_forest, neural_network
- cross_validation, grid_search, feature_engineering
- regularization, ensemble, stacking
- dimensionality_reduction, clustering

For key_decisions, identify important design choices:
- Model selection rationale
- Feature engineering approach
- Validation strategy
- Hyperparameter tuning method

For complexity:
- simple: Basic model, straightforward preprocessing
- moderate: Multiple models or advanced preprocessing
- complex: Ensemble, extensive feature engineering, or custom architectures

For data_flow:
- Briefly describe how data moves through the code (load → preprocess → train → evaluate)
"""
    
    def _fallback_analysis(self, code: str) -> CodeAnalysis:
        """Fallback analysis if LLM fails."""
        # Extract imports
        import_lines = [line for line in code.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')]
        libraries = []
        for line in import_lines:
            parts = line.split()
            if 'import' in parts:
                idx = parts.index('import')
                if idx + 1 < len(parts):
                    lib = parts[idx + 1].split('.')[0]
                    libraries.append(lib)
        
        # Common ML techniques
        techniques = []
        code_lower = code.lower()
        technique_keywords = {
            "gradient_boosting": ["gradientboosting", "xgboost", "lgbm", "catboost"],
            "random_forest": ["randomforest"],
            "neural_network": ["neural", "keras", "torch", "tensorflow"],
            "cross_validation": ["cross_val", "kfold"],
            "feature_engineering": ["featureengineering", "feature_union"]
        }
        
        for technique, keywords in technique_keywords.items():
            if any(kw in code_lower for kw in keywords):
                techniques.append(technique)
        
        # Assess complexity
        lines = len(code.split('\n'))
        if lines < 50:
            complexity = "simple"
        elif lines < 150:
            complexity = "moderate"
        else:
            complexity = "complex"
        
        return CodeAnalysis(
            techniques=techniques,
            key_decisions=[],
            libraries_used=libraries,
            complexity=complexity,
            data_flow_summary="Data is loaded, processed, and used for model training."
        )

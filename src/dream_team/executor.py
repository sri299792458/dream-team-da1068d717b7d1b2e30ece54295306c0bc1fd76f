"""
Code execution module for Dream Team framework.

Provides safe execution environment for agent-generated code.
"""

import sys
import io
import traceback
import subprocess
import re
import importlib
import json
import os
import signal
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import torch
from pathlib import Path


class TimeoutError(Exception):
    """Raised when code execution exceeds timeout"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Code execution exceeded timeout limit")


class CodeExecutor:
    """Executes Python code in a controlled environment."""

    def __init__(self, data_context: Dict[str, Any] = None, auto_install: bool = True, max_output_length: int = 10000):
        """
        Initialize executor with data context.

        Args:
            data_context: Dictionary of data/variables available to executed code
            auto_install: Whether to automatically install missing packages (default: True)
            max_output_length: Maximum length of output to store (default: 10000 chars)
        """
        self.data_context = data_context or {}
        self.execution_history = []
        self.auto_install = auto_install
        self.installed_packages = set()
        self.max_output_length = max_output_length

        # Setup workspace directory
        self.workspace_dir = Path(os.getcwd()) / "workspace"
        self.workspace_dir.mkdir(exist_ok=True)

    def execute(
        self,
        code: str,
        description: str = "",
        timeout: int = 1800
    ) -> Dict[str, Any]:
        """
        Execute Python code and return results.

        Args:
            code: Python code to execute
            description: Description of what this code does
            timeout: Timeout in seconds (enforced on Unix/Linux/Mac; gracefully degrades on Windows)

        Returns:
            {
                'success': bool,
                'output': stdout output,
                'error': error message if failed,
                'variables': dict of new variables created,
                'metrics': extracted metrics if any
            }
        """
        print(f"\n⚙️ Executing: {description or 'Code block'}")

        # Capture stdout
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # Prepare execution environment with SINGLE namespace
        exec_namespace = {
            'pd': pd,
            'np': np,
            'torch': torch,
            'Path': Path,
            '__builtins__': __builtins__,
        }

        # Add data context to namespace
        exec_namespace.update(self.data_context)

        # Track initial keys to identify new variables
        initial_keys = set(exec_namespace.keys())

        result = {
            'success': False,
            'output': '',
            'error': None,
            'variables': {},
            'metrics': {},
            'code': code,
            'description': description
        }

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Set up timeout (Unix/Linux/Mac only)
            timeout_set = False
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
                timeout_set = True

            try:
                # Execute code with single namespace
                exec(code, exec_namespace)
                result['success'] = True
            finally:
                # Cancel timeout if it was set
                if timeout_set:
                    signal.alarm(0)

            full_output = stdout_capture.getvalue()
            result['output'] = self._truncate_output(full_output)
            if len(full_output) > self.max_output_length:
                result['output_truncated'] = True
                result['original_output_length'] = len(full_output)

            # Extract NEW variables created during execution
            result['variables'] = {
                k: v for k, v in exec_namespace.items()
                if k not in initial_keys and not k.startswith('_')
            }

            # Extract metrics
            metric_names = ['mae', 'rmse', 'f1', 'accuracy', 'score', 'cv_scores', 'error', 'loss', 'val_loss', 'auc', 'precision', 'recall']
            result['metrics'] = {
                k: v for k, v in exec_namespace.items()
                if any(metric in k.lower() for metric in metric_names) and isinstance(v, (int, float, np.number))
            }

            truncation_note = ""
            if result.get('output_truncated'):
                truncation_note = f" (output truncated: {result['original_output_length']} → {len(result['output'])} chars)"
            print(f"   ✅ Success{truncation_note}")
            if result['metrics']:
                print(f"   📊 Metrics: {result['metrics']}")

        except TimeoutError as e:
            result['success'] = False
            result['error'] = f"Execution timeout ({timeout}s exceeded). Code likely has infinite loop or very long computation."
            result['traceback'] = traceback.format_exc()

            full_output = stdout_capture.getvalue()
            result['output'] = self._truncate_output(full_output)
            if len(full_output) > self.max_output_length:
                result['output_truncated'] = True
                result['original_output_length'] = len(full_output)

            print(f"   ⏱️  Timeout: Code execution exceeded {timeout}s limit")

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()

            full_output = stdout_capture.getvalue()
            result['output'] = self._truncate_output(full_output)
            if len(full_output) > self.max_output_length:
                result['output_truncated'] = True
                result['original_output_length'] = len(full_output)

            print(f"   ❌ Error: {e}")

            # Check if it's a ModuleNotFoundError and auto-install is enabled
            if self.auto_install:
                missing_package = self._extract_missing_module(str(e), result['traceback'])
                if missing_package:
                    result['missing_package'] = missing_package

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Update data context with new variables
        if result['success']:
            self.data_context.update(result['variables'])

        # Store in history
        self.execution_history.append(result)

        return result

    def execute_with_retry(
        self,
        code: str,
        description: str = "",
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Execute code with automatic retry on failure.

        If failure is due to missing package, automatically install and retry.

        Returns the result of first successful execution or last failure.
        """
        for attempt in range(max_retries + 1):
            result = self.execute(code, description)
            if result['success']:
                return result

            # Check if failure was due to missing package
            if 'missing_package' in result and self.auto_install:
                package = result['missing_package']
                if self._install_package(package):
                    print(f"   🔄 Retrying after installing {package}...")
                    continue

            if attempt < max_retries:
                print(f"   🔄 Retry {attempt + 1}/{max_retries}")

        return result

    def get_variable(self, name: str) -> Any:
        """Get a variable from the execution context"""
        return self.data_context.get(name)

    def set_variable(self, name: str, value: Any):
        """Set a variable in the execution context"""
        self.data_context[name] = value

    def _truncate_output(self, output: str) -> str:
        """
        Truncate output to max_output_length, keeping most recent content.
        """
        if len(output) <= self.max_output_length:
            return output

        # Keep first 2000 chars (initial output) and last N chars (final results)
        first_chunk_size = 2000
        last_chunk_size = self.max_output_length - first_chunk_size - 100

        first_chunk = output[:first_chunk_size]
        last_chunk = output[-last_chunk_size:]

        truncated_lines = output[first_chunk_size:-last_chunk_size].count('\n')

        return (
            f"{first_chunk}\n"
            f"\n... [Truncated {truncated_lines} lines of verbose output] ...\n\n"
            f"{last_chunk}"
        )

    def get_metrics_history(self) -> list:
        """Extract all metrics from execution history"""
        metrics = []
        for execution in self.execution_history:
            if execution['metrics']:
                metrics.append({
                    'description': execution['description'],
                    'metrics': execution['metrics']
                })
        return metrics

    def clear_history(self):
        """Clear execution history"""
        self.execution_history = []

    def _install_package(self, package_name: str) -> bool:
        """
        Install a Python package using pip.
        """
        if package_name in self.installed_packages:
            print(f"   📦 {package_name} already installed this session")
            return True

        print(f"   📦 Installing missing package: {package_name}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            importlib.invalidate_caches()
            self.installed_packages.add(package_name)
            print(f"   ✅ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package_name}: {e}")
            return False

    def _extract_missing_module(self, error_msg: str, traceback_str: str) -> Optional[str]:
        """
        Extract the missing module name from a ModuleNotFoundError.
        """
        if "No module named" not in error_msg:
            return None

        match = re.search(r"No module named ['\"]([^'\"\.]+)", error_msg)
        if match:
            module_name = match.group(1)

            # Map common module names to package names
            package_map = {
                'sklearn': 'scikit-learn',
                'cv2': 'opencv-python',
                'PIL': 'Pillow',
                'skopt': 'scikit-optimize',
            }

            return package_map.get(module_name, module_name)

        return None

    def summary(self) -> str:
        """Get execution summary"""
        total = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e['success'])
        failed = total - successful

        return f"Executions: {total} total, {successful} successful, {failed} failed"


def extract_code_from_text(text: str) -> str:
    """
    Extract Python code from markdown code blocks or plain text.
    Prefer the last code block (often the final full script).
    """
    if "```" in text:
        parts = text.split("```")
        chosen = None
        for i, part in enumerate(parts):
            if part.startswith("python\n") or part.startswith("python "):
                # strip "python" and keep the rest
                chosen = part[6:]
            elif i % 2 == 1:  # odd indices are code blocks without language tag
                chosen = part
        if chosen is not None:
            return chosen.strip()

    return text.strip()


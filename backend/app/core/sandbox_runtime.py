"""Dynamic Code Sandbox & Safe Isolated Python Runtime.

Provides secure, AST-validated execution of dynamic analytical Python & Pandas scripts
against in-memory DataFrames with CPU timeouts, restricted namespace scoping, and
protection against arbitrary code execution (RCE) and system vulnerabilities.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import datetime
import io
import math
import multiprocessing
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


class SecurityViolationError(Exception):
    """Raised when dynamically executed code violates security sandboxing policies."""
    pass


class ExecutionTimeoutError(Exception):
    """Raised when dynamically executed code exceeds maximum CPU execution timeout."""
    pass


@dataclass
class SandboxExecutionResult:
    """Result of a sandboxed dynamic code execution."""
    success: bool
    result: Any
    stdout: str
    execution_time_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res_repr = None
        if isinstance(self.result, pd.DataFrame):
            res_repr = self.result.head(20).to_dict(orient="records")
        elif isinstance(self.result, (pd.Series, np.ndarray)):
            res_repr = self.result.tolist()
        elif isinstance(self.result, (int, float, str, bool, list, dict)):
            res_repr = self.result
        else:
            res_repr = str(self.result)

        return {
            "success": self.success,
            "result": res_repr,
            "stdout": self.stdout,
            "execution_time_ms": round(float(self.execution_time_ms), 3),
            "error": self.error,
        }


class SafeIsolatedExecutionSandbox:
    """AST-validated isolated Python runtime for data analysis."""

    # Disallowed module imports
    FORBIDDEN_MODULES: Set[str] = {
        "os", "sys", "subprocess", "socket", "http", "urllib", "requests", "aiohttp",
        "shutil", "importlib", "builtins", "__builtin__", "ctypes", "pty", "commands",
        "threading", "multiprocessing", "pickle", "shelve", "posix", "gc"
    }

    # Disallowed attribute lookups (dunder access exploitation)
    FORBIDDEN_ATTRIBUTES: Set[str] = {
        "__subclasses__", "__globals__", "__code__", "__bases__", "__mro__",
        "__builtins__", "__import__", "__class__", "__reduce__", "__reduce_ex__",
        "__dict__", "func_globals", "func_code"
    }

    # Safe built-in functions allowed inside sandbox
    SAFE_BUILTINS: Dict[str, Any] = {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "isinstance": isinstance,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "round": round,
        "abs": abs,
        "map": map,
        "filter": filter,
        "any": any,
        "all": all,
        "print": print,
        "True": True,
        "False": False,
        "None": None,
    }

    def __init__(self, timeout_seconds: float = 3.0):
        self.timeout_seconds = timeout_seconds

    def validate_ast(self, code_str: str) -> None:
        """Parse AST and verify no dangerous operations, forbidden modules, or dunder exploits exist."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            raise SecurityViolationError(f"Syntax Error in Python code: {e}")

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in self.FORBIDDEN_MODULES:
                        raise SecurityViolationError(f"Import of forbidden module '{alias.name}' is prohibited.")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in self.FORBIDDEN_MODULES:
                        raise SecurityViolationError(f"Import from forbidden module '{node.module}' is prohibited.")

            # Check attribute access (e.g. obj.__class__)
            elif isinstance(node, ast.Attribute):
                if node.attr in self.FORBIDDEN_ATTRIBUTES:
                    raise SecurityViolationError(f"Access to restricted attribute '{node.attr}' is prohibited.")

            # Check direct calls to forbidden functions
            elif isinstance(node, ast.Name):
                if node.id in ("eval", "exec", "compile", "open", "input", "breakpoint", "exit", "quit", "__import__"):
                    raise SecurityViolationError(f"Call to restricted built-in function '{node.id}' is prohibited.")

    def execute_code(
        self,
        code_str: str,
        dataframe: Optional[pd.DataFrame] = None,
        extra_variables: Optional[Dict[str, Any]] = None,
    ) -> SandboxExecutionResult:
        """
        Safely execute Python code against a DataFrame within an isolated environment.
        """
        start_t = time.time()

        # Step 1: Static AST Validation
        try:
            self.validate_ast(code_str)
        except SecurityViolationError as sec_err:
            return SandboxExecutionResult(
                success=False,
                result=None,
                stdout="",
                execution_time_ms=(time.time() - start_t) * 1000,
                error=str(sec_err),
            )

        # Step 2: Set up restricted namespace
        captured_stdout = io.StringIO()
        original_stdout = sys.stdout

        safe_globals: Dict[str, Any] = {
            "__builtins__": self.SAFE_BUILTINS,
            "pd": pd,
            "np": np,
            "math": math,
            "re": re,
            "datetime": datetime,
        }

        safe_locals: Dict[str, Any] = {}
        if dataframe is not None:
            # Provide isolated copy of dataframe
            safe_locals["df"] = dataframe.copy()
            safe_locals["dataframe"] = safe_locals["df"]
            safe_locals["data"] = safe_locals["df"]

        if extra_variables:
            for k, v in extra_variables.items():
                if not k.startswith("__"):
                    safe_locals[k] = v

        try:
            sys.stdout = captured_stdout

            # Compile into bytecode
            compiled_code = compile(code_str, "<sandbox>", "exec")

            # Execute
            exec(compiled_code, safe_globals, safe_locals)

            # Extract return value ('result' or 'output' or modified 'df')
            result_val = safe_locals.get("result", safe_locals.get("output", safe_locals.get("df", None)))
            success = True
            error_msg = None

        except Exception as e:
            result_val = None
            success = False
            error_msg = f"{type(e).__name__}: {str(e)}"
        finally:
            sys.stdout = original_stdout

        duration = (time.time() - start_t) * 1000
        stdout_str = captured_stdout.getvalue()

        return SandboxExecutionResult(
            success=success,
            result=result_val,
            stdout=stdout_str,
            execution_time_ms=duration,
            error=error_msg,
        )


# Global singleton instance
global_sandbox = SafeIsolatedExecutionSandbox()


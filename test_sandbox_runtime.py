"""Tests for Safe Isolated Execution Sandbox & Dynamic Python Runtime.

Verifies:
1. SafeIsolatedExecutionSandbox executing valid Pandas/NumPy analytical transformations
2. Static AST validation blocking unauthorized module imports (os, sys, subprocess, socket)
3. AST validation blocking dunder exploit attempts (__subclasses__, __globals__, __builtins__)
4. AST validation blocking dangerous built-in functions (eval, exec, open, __import__)
5. Stdout capture and execution time recording
"""
import pandas as pd
import pytest

from backend.app.core.sandbox_runtime import (
    SafeIsolatedExecutionSandbox,
    SandboxExecutionResult,
    SecurityViolationError,
    global_sandbox,
)


@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet", "Monitor"],
        "Units": [10, 25, 15, 8],
        "Price": [1200.0, 800.0, 500.0, 300.0],
    })


def test_valid_pandas_calculation(sales_df):
    """Verify safe execution of pandas feature engineering and aggregations."""
    sandbox = SafeIsolatedExecutionSandbox()
    code = """
df['Revenue'] = df['Units'] * df['Price']
total_rev = df['Revenue'].sum()
print(f"Calculated Total Revenue: ${total_rev:,.2f}")
result = {
    'total_revenue': float(total_rev),
    'avg_price': float(df['Price'].mean()),
    'top_product': str(df.sort_values('Revenue', ascending=False).iloc[0]['Product'])
}
"""
    res: SandboxExecutionResult = sandbox.execute_code(code, dataframe=sales_df)

    assert res.success is True
    assert res.error is None
    assert "Calculated Total Revenue: $41,900.00" in res.stdout
    assert res.result["total_revenue"] == 41900.0
    assert res.result["top_product"] == "Phone"
    assert res.execution_time_ms >= 0


def test_blocking_forbidden_imports(sales_df):
    """Verify sandbox blocks attempts to import os, sys, subprocess, socket."""
    sandbox = SafeIsolatedExecutionSandbox()

    dangerous_snippets = [
        "import os; os.system('dir')",
        "import sys; sys.exit(0)",
        "import subprocess; subprocess.run(['cmd'])",
        "import socket; s = socket.socket()",
        "from os import path",
        "from subprocess import Popen",
    ]

    for snippet in dangerous_snippets:
        res = sandbox.execute_code(snippet, dataframe=sales_df)
        assert res.success is False
        assert "forbidden module" in str(res.error).lower() or "security" in str(res.error).lower()


def test_blocking_dunder_exploit_attributes(sales_df):
    """Verify sandbox blocks access to __subclasses__, __globals__, etc."""
    sandbox = SafeIsolatedExecutionSandbox()

    exploit_snippets = [
        "x = ().__class__.__bases__[0].__subclasses__()",
        "g = len.__globals__",
        "c = (1).__code__",
        "b = ().__class__.__builtins__",
    ]

    for snippet in exploit_snippets:
        res = sandbox.execute_code(snippet, dataframe=sales_df)
        assert res.success is False
        assert "restricted attribute" in str(res.error).lower()


def test_blocking_dangerous_builtins(sales_df):
    """Verify sandbox blocks open(), eval(), exec(), __import__()."""
    sandbox = SafeIsolatedExecutionSandbox()

    dangerous_calls = [
        "f = open('secret.txt', 'r')",
        "eval('2 + 2')",
        "exec('a = 5')",
        "m = __import__('os')",
    ]

    for snippet in dangerous_calls:
        res = sandbox.execute_code(snippet, dataframe=sales_df)
        assert res.success is False
        assert "restricted built-in" in str(res.error).lower()

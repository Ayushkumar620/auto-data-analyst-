"""Rewrite agent/planner.py from the pristine git-HEAD version, applying the
three ResultValidator wiring changes via exact-anchor str.replace with
assertions, so the editor's whitespace-matching bugs can't corrupt indentation.
"""
import ast
import subprocess
import sys

try:
    original = subprocess.check_output(
        ["git", "show", "HEAD:agent/planner.py"], text=True, stderr=subprocess.STDOUT
    )
except subprocess.CalledProcessError as e:
    print("GIT FAIL:", getattr(e, "output", e))
    sys.exit(1)

src = original

A1 = 'import time\n\nfrom .agents import (\n    DataLoadingAgent,'
N1 = ('import time\n'
      'from typing import Any, Dict, Optional\n\n'
      'import pandas as pd\n\n'
      'from .result_validator import ResultValidator\n'
      'from .agents import (\n    DataLoadingAgent,')

A2 = '    def __init__(self, data=None):\n        self.data = data\n'
N2 = ('    def __init__(self, data=None, validator: ResultValidator = None):\n'
      '        self.data = data\n'
      '        self._validator = validator or ResultValidator()\n')

A3 = ('        agent._start()\n'
      '        try:\n'
      '            result = agent.run(task)\n'
      '            return result\n'
      '        except Exception as e:\n'
      '            return agent._error(str(e))\n'
      '\n'
      '    def run_pipeline(self, data=None, steps=None):')
N3 = ('        agent._start()\n'
      '        try:\n'
      '            result = agent.run(task)\n'
      '        except Exception as e:\n'
      '            result = agent._error(str(e))\n'
      '        context = self._validation_context(data or self.data)\n'
      '        # Validate + repair every AgentResult before it leaves the planner so\n'
      '        # downstream consumers never see an unvalidated/unrepaired result.\n'
      '        self._validator.repair(result, context)\n'
      '        return result\n'
      '\n'
      '    @staticmethod\n'
      '    def _validation_context(\n'
      '        data: Any,\n'
      '    ) -> Optional[Dict[str, Any]]:\n'
      '        """Build the validation context the ResultValidator uses to cross-check\n'
      '        evidence against the real dataset."""\n'
      '        if data is None:\n'
      '            return None\n'
      '        if isinstance(data, pd.DataFrame):\n'
      '            return {\n'
      '                "dataframe": data,\n'
      '                "columns": [str(c) for c in data.columns],\n'
      '                "row_count": len(data),\n'
      '            }\n'
      '        if isinstance(data, dict):\n'
      '            frames = [f for f in data.values() if isinstance(f, pd.DataFrame)]\n'
      '            columns = sorted({str(c) for f in frames for c in f.columns})\n'
      '            row_count = max((len(f) for f in frames), default=0)\n'
      '            return {"dataframe": data, "columns": columns, "row_count": row_count}\n'
      '        return None\n'
      '\n'
      '    def run_pipeline(self, data=None, steps=None):')

for name, a, n in [("imports", A1, N1), ("init", A2, N2), ("run_agent", A3, N3)]:
    cnt = src.count(a)
    if cnt != 1:
        print(f"ANCHOR {name!r} matched {cnt} times (expected 1)")
        sys.exit(1)
    src = src.replace(a, n)

# corruption guards
assert "                agent._start()" not in src
assert "        def __init__(self, data=None, validator: ResultValidator = None):\n            self.data" not in src
# positive checks
assert "from .result_validator import ResultValidator" in src
assert "self._validator = validator or ResultValidator()" in src
assert "self._validator.repair(result, context)" in src
assert "def _validation_context(" in src

open("agent/planner.py", "w", encoding="utf-8").write(src)
ast.parse(src)
print("planner.py rewritten & AST OK:", len(src), "chars")

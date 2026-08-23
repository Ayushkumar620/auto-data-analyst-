import ast

p = r"agent\planner.py"
src = open(p, encoding="utf-8").read()

bad = "context = self._validation_context(data or self.data)"
good = "context = self._validation_context(data)"
assert src.count(bad) == 1, "bad context line count %d" % src.count(bad)
src = src.replace(bad, good)

assert "data or self.data" not in src, "data or self.data still present"
ast.parse(src)
open(p, "w", encoding="utf-8").write(src)
print("planner.py patched: data or self.data -> data ; AST OK")

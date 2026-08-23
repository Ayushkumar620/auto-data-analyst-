import ast

src = open(r"agent\planner.py", encoding="utf-8").read()

b1 = '        def __init__(self, data=None, validator: ResultValidator = None):'
g1 = '    def __init__(self, data=None, validator: ResultValidator = None):'

b2 = '                agent._start()'
g2 = '        agent._start()'

assert src.count(b1) == 1, "init count %d" % src.count(b1)
assert src.count(b2) == 1, "start count %d" % src.count(b2)

src = src.replace(b1, g1).replace(b2, g2)
ast.parse(src)
open(r"agent\planner.py", "w", encoding="utf-8").write(src)
print("FIXED: AST OK")

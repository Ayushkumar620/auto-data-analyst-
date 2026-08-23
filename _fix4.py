import ast

p = r"agent\result_validator.py"
src = open(p, encoding="utf-8").read()

old = (
    "        ref = evidence.data_ref or {}\n"
    "        names = ref.get(\"column_names\") or ref.get(\"columns\") or []\n"
    "        if not isinstance(names, (list, tuple)):\n"
    "            names = [ref.get(\"column\")] if ref.get(\"column\") else []\n"
    "        return any(isinstance(n, str) and n not in real for n in names)\n"
)
new = (
    "        ref = evidence.data_ref or {}\n"
    "        names = ref.get(\"column_names\") or ref.get(\"columns\") or []\n"
    "        if isinstance(names, str):\n"
    "            names = [names]\n"
    "        elif not isinstance(names, (list, tuple)):\n"
    "            names = []\n"
    "        single_col = ref.get(\"column\") or ref.get(\"col\")\n"
    "        if isinstance(single_col, str):\n"
    "            names = list(names) + [single_col]\n"
    "        return any(isinstance(n, str) and n not in real for n in names)\n"
)

assert src.count(old) == 1, "anchor count %d" % src.count(old)
src = src.replace(old, new)
ast.parse(src)
open(p, "w", encoding="utf-8").write(src)
print("result_validator.py patched: _references_unknown_column now checks column/col ; AST OK")

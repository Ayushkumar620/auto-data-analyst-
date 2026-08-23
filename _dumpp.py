path = r"agent\planner.py"
lines = open(path, encoding="utf-8").read().split("\n")
for i, ln in enumerate(lines, 1):
    print(f"{i:>4}|{ln}")

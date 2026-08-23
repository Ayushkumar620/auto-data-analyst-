path = r"agent\planner.py"
lines = open(path, encoding="utf-8").read().split("\n")
for i in range(116, 146):
    print(f"{i+1:>4}|{lines[i]}")

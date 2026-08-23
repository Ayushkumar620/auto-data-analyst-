for label, path, a, b in [
    ("planner.py run_agent/init (117-150)", "agent/planner.py", 117, 150),
    ("result_validator.py _references_unknown_column (~380-412)", "agent/result_validator.py", 380, 412),
]:
    lines = open(path, encoding="utf-8").read().split("\n")
    print("==== %s ====" % label)
    for i in range(max(0, a - 1), min(len(lines), b)):
        print(f"{i+1:>4}|{lines[i]}")
    print()

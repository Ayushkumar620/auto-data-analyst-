import sys

lines = open("agent/result_validator.py").read().splitlines()
with open("_dump.txt", "w") as out:
    for i in range(268, 296):
        out.write(f"{i + 1}: {lines[i]!r}\n")
    # also the three def-line locations
    for needle in ("def _evidence_check", "def _cross_check(self", "def _cross_check_columns"):
        for j, l in enumerate(lines):
            if needle in l:
                out.write(f"DEF {j + 1}: {l!r}\n")
                break
print("wrote _dump.txt")

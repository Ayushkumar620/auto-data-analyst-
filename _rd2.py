data = open("_p.txt", encoding="utf-8", errors="ignore").read()
lines = data.splitlines()
print("TOTAL LINES:", len(lines))
for ln in lines[-40:]:
    print(repr(ln[:200]))

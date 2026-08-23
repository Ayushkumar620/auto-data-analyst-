data = open("_p.txt", encoding="utf-8", errors="ignore").read()
for ln in data.splitlines():
    if any(k in ln for k in ("passed", "failed", "error", "FAILED", "ValueError", "assert", "E   ", "E ")):
        print(repr(ln[:160]))

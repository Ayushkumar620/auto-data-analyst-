p = "docs/DEVELOPMENT_STATUS.md"
s = open(p, encoding="utf-8").read()

old4 = "| 4. **Create `ResultValidator`** - Validates schema, cross-checks data, verifies evidence (also fixes the predictor classification `AttributeError` via repair)"
new4 = ("| 4. ✅ **Create `ResultValidator` + validation gate in PlannerAgent** - Done. "
        "`validate()` runs 5 checks (schema, consistency, evidence, claim integrity, data cross-check); "
        "`repair()` clamps confidence, stamps timestamps, drops invalid / unknown-column evidence. "
        "`_references_unknown_column` fixed to inspect single `column`/`col` refs. Wired into "
        "`PlannerAgent.run_agent` via `ResultValidator.repair(result, context)` so every `AgentResult` "
        "is validated/repaired before downstream use.")
assert s.count(old4) == 1, "task4 line %d" % s.count(old4)
s = s.replace(old4, new4)

cl2 = "| 2026-08-23 | **Task 2: Evidence Model Integration** | `agent/base.py`, `agent/agents.py` | 73 unit tests pass; all 8 agents produce evidence-carrying `AgentResult` |"
cl4 = "| 2026-08-23 | **Task 4: Result Validator + planner wiring** | `agent/result_validator.py`, `agent/planner.py`, `test_planner_validation.py` | 26 root tests pass (20 validator + 6 chat/pipeline/planner); no regressions |"
assert s.count(cl2) == 1, "changelog task2 %d" % s.count(cl2)
s = s.replace(cl2, cl2 + "\n" + cl4)

oldrec = "## Recommendation: Start With Task #1\n\n**Implement `AgentResult` / `AgentError` / `BaseAgent` contract first.**"
newrec = ("## Recommendation: Start With Task #4 (Reliability Gates)\n\n"
          "**Tasks 1-3 are complete** (AgentResult/AgentError/BaseAgent contracts, evidence on all 8 agents). "
          "The reliability gate (Task 4) is now done; the next priorities are the retry/repair loop "
          "(`agent/retry.py`, Task 6) and populating the shared `DatasetKnowledge` object (Task 5).")
assert s.count(oldrec) == 1, "recommendation %d" % s.count(oldrec)
s = s.replace(oldrec, newrec)

open(p, "w", encoding="utf-8").write(s)
print("DEVELOPMENT_STATUS.md updated OK")

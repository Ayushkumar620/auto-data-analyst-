"""
Execution Engine for Dynamic Dependency-Aware Execution Graphs.

Executes an ExecutionPlan step-by-step:
- Topological dependency resolution
- Independent parallel execution where safe
- Output propagation (e.g., Cleaned DataFrame -> Downstream Aggregations)
- Recoverable failure retries and error isolation
- Graceful partial success handling (preserves successful outputs when independent tasks fail)
- Standardized AgentResult validation and execution tracing
"""
from __future__ import annotations

import concurrent.futures
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from agent.schemas import AgentError, AgentResult, AgentStatus, ClaimType, Evidence
from agent.tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ExecutionEngine:
    """
    Dependency-aware, high-concurrency execution engine for analytical task plans.
    Supports topological level grouping with parallel execution for independent steps.
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None, max_workers: int = 4):
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY
        self.max_workers = max_workers

    def _build_dependency_levels(self, steps: List[Any]) -> List[List[Any]]:
        """
        Group steps into topological dependency levels where all steps in level L_k
        only depend on steps in levels < k and can be safely executed concurrently.
        """
        if not steps:
            return []

        step_map = {str(getattr(s, "step_id", idx + 1)): s for idx, s in enumerate(steps)}
        resolved_step_ids: Set[str] = set()
        remaining_steps = list(steps)
        levels: List[List[Any]] = []

        while remaining_steps:
            current_level: List[Any] = []
            next_remaining: List[Any] = []

            for step in remaining_steps:
                s_id = str(getattr(step, "step_id", ""))
                deps = [str(d) for d in getattr(step, "dependencies", [])]

                # If all dependencies have been resolved in earlier levels, include in current level
                if all(d in resolved_step_ids for d in deps):
                    current_level.append(step)
                else:
                    next_remaining.append(step)

            # Safeguard against circular dependencies: if no step can be resolved, take the first one
            if not current_level and next_remaining:
                current_level.append(next_remaining.pop(0))

            for s in current_level:
                resolved_step_ids.add(str(getattr(s, "step_id", "")))

            levels.append(current_level)
            remaining_steps = next_remaining

        return levels

    def _execute_single_step(
        self,
        step: Any,
        current_data: pd.DataFrame,
        step_results: Dict[str, AgentResult],
    ) -> Tuple[Any, Optional[AgentResult], List[str], List[AgentError], float, str]:
        """Execute a single step with retry policy and return diagnostic output."""
        s_id = str(getattr(step, "step_id", ""))
        s_name = getattr(step, "name", getattr(step, "purpose", f"Step {s_id}"))
        tool_name = getattr(step, "tool_name", getattr(step, "agent_class_name", ""))
        dependencies = [str(d) for d in getattr(step, "dependencies", [])]

        step_warnings: List[str] = []
        step_errors: List[AgentError] = []

        # 1. Check if any upstream prerequisite dependency failed or was skipped
        for dep_id in dependencies:
            dep_res = step_results.get(dep_id)
            if not dep_res or not dep_res.is_success:
                setattr(step, "status", StepStatus.SKIPPED.value)
                step_warnings.append(f"Step {s_id} ('{s_name}') skipped due to upstream dependency failure.")
                return step, None, step_warnings, step_errors, 0.0, StepStatus.SKIPPED.value

        # 2. Execute step with retry policy
        setattr(step, "status", StepStatus.RUNNING.value)
        step_start = datetime.now()
        retry_policy = getattr(step, "retry_policy", {"max_retries": 1})
        max_retries = retry_policy.get("max_retries", 1) if isinstance(retry_policy, dict) else 1

        result: Optional[AgentResult] = None
        attempt = 0

        while attempt <= max_retries:
            try:
                params = getattr(step, "inputs", getattr(step, "parameters", {}))
                task_inputs = {"data": current_data, **params}

                if tool_name.lower() in ("reporting", "report", "reportagent", "insightagent", "decisionexplainer") or "agent_outputs" in params:
                    task_inputs["agent_outputs"] = list(step_results.values())

                if self.tool_registry.has_tool(tool_name):
                    res = self.tool_registry.execute(tool_name, **task_inputs)
                    if isinstance(res, AgentResult):
                        result = res
                    else:
                        result = AgentResult.success(output={"result": res}, agent_name=tool_name)
                elif tool_name.lower() in ("analysis", "analysisagent", "summary", "eda"):
                    from agent.agents import AnalysisAgent
                    result = AnalysisAgent().execute(task_inputs)
                else:
                    result = AgentResult.error(
                        error=f"Tool '{tool_name}' is not registered.",
                        code="TOOL_NOT_FOUND",
                        agent_name=tool_name or "ExecutionEngine",
                    )

                if result is not None and result.is_success:
                    break
                else:
                    attempt += 1
                    if attempt <= max_retries:
                        setattr(step, "status", StepStatus.RETRYING.value)
                        step_warnings.append(f"Step {s_id} retry {attempt}/{max_retries}...")
                        time.sleep(0.02)
            except Exception as exc:
                attempt += 1
                if attempt > max_retries:
                    result = AgentResult.error(
                        error=f"Step execution failed: {str(exc)}",
                        agent_name=tool_name or "ExecutionEngine",
                    )
                    break
                time.sleep(0.02)

        duration_ms = round((datetime.now() - step_start).total_seconds() * 1000, 2)
        setattr(step, "duration_ms", duration_ms)
        setattr(step, "result", result)

        if result is not None and result.is_success:
            status_val = StepStatus.SUCCESS.value
        else:
            status_val = StepStatus.FAILED.value
            if result:
                step_errors.extend(result.errors)
                step_warnings.extend(result.warnings)
            step_warnings.append(f"Step {s_id} ('{s_name}') failed after {attempt} attempts.")

        setattr(step, "status", status_val)
        return step, result, step_warnings, step_errors, duration_ms, status_val

    def execute_plan(
        self,
        plan: Any,
        dataframe: pd.DataFrame,
    ) -> AgentResult:
        """
        Execute an ExecutionPlan (or legacy TaskPlan) with level-based concurrency,
        dependency tracking, retry policy, and partial success isolation.
        """
        start_time = datetime.now()
        current_data = dataframe if dataframe is not None else pd.DataFrame()
        step_results: Dict[str, AgentResult] = {}
        all_evidence: List[Evidence] = []
        all_warnings: List[str] = []
        all_errors: List[AgentError] = []
        execution_trace: List[Dict[str, Any]] = []

        steps = plan.steps if hasattr(plan, "steps") else []
        levels = self._build_dependency_levels(steps)

        # Execute level by level: steps within the same level can run concurrently
        for level in levels:
            if len(level) == 1:
                step = level[0]
                _, result, s_warn, s_err, dur, s_status = self._execute_single_step(
                    step, current_data, step_results
                )
                all_warnings.extend(s_warn)
                all_errors.extend(s_err)
                s_id = str(getattr(step, "step_id", ""))
                tool_name = getattr(step, "tool_name", getattr(step, "agent_class_name", ""))
                s_name = getattr(step, "name", getattr(step, "purpose", f"Step {s_id}"))

                if result is not None and result.is_success:
                    step_results[s_id] = result
                    all_evidence.extend(result.evidence)
                    all_warnings.extend(result.warnings)

                    # Carry cleaned data forward if cleaning occurred
                    if tool_name in ("data_cleaning", "CleaningAgent"):
                        reports = result.output.get("reports", []) if isinstance(result.output, dict) else []
                        if reports and isinstance(reports, list) and "cleaned_data" in reports[0]:
                            current_data = pd.DataFrame(reports[0]["cleaned_data"])
                        elif isinstance(result.data, dict) and "cleaned_data" in result.data:
                            current_data = pd.DataFrame(result.data["cleaned_data"])

                execution_trace.append({
                    "step_id": s_id,
                    "name": s_name,
                    "tool": tool_name,
                    "status": s_status,
                    "duration_ms": dur,
                    "confidence": getattr(result, "confidence", 0.0) if result else 0.0,
                })
            else:
                # Multiple independent steps in this level: execute concurrently
                workers = min(len(level), self.max_workers)
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_step = {
                        executor.submit(self._execute_single_step, s, current_data, dict(step_results)): s
                        for s in level
                    }
                    for future in concurrent.futures.as_completed(future_to_step):
                        step, result, s_warn, s_err, dur, s_status = future.result()
                        all_warnings.extend(s_warn)
                        all_errors.extend(s_err)
                        s_id = str(getattr(step, "step_id", ""))
                        tool_name = getattr(step, "tool_name", getattr(step, "agent_class_name", ""))
                        s_name = getattr(step, "name", getattr(step, "purpose", f"Step {s_id}"))

                        if result is not None and result.is_success:
                            step_results[s_id] = result
                            all_evidence.extend(result.evidence)
                            all_warnings.extend(result.warnings)

                        execution_trace.append({
                            "step_id": s_id,
                            "name": s_name,
                            "tool": tool_name,
                            "status": s_status,
                            "duration_ms": dur,
                            "confidence": getattr(result, "confidence", 0.0) if result else 0.0,
                        })

        total_duration = round((datetime.now() - start_time).total_seconds() * 1000, 2)
        if hasattr(plan, "total_duration_ms"):
            plan.total_duration_ms = total_duration

        # Determine Overall Execution Status
        total_steps = len(steps)
        success_count = sum(1 for s in steps if getattr(s, "status", "") == StepStatus.SUCCESS.value)
        failed_count = sum(1 for s in steps if getattr(s, "status", "") in (StepStatus.FAILED.value, StepStatus.SKIPPED.value))

        if success_count == total_steps and total_steps > 0:
            final_status = AgentStatus.COMPLETED
            msg = f"Execution plan completed successfully ({success_count}/{total_steps} steps executed)."
        elif success_count > 0:
            final_status = AgentStatus.PARTIAL
            msg = f"Execution plan completed with partial success ({success_count} succeeded, {failed_count} failed/skipped)."
        else:
            final_status = AgentStatus.ERROR
            msg = f"Execution plan failed ({failed_count}/{total_steps} steps failed)."

        step_outputs = {sid: res.output for sid, res in step_results.items()}
        avg_confidence = round(
            float(sum(getattr(res, "confidence", 1.0) for res in step_results.values()) / len(step_results)), 4
        ) if step_results else 0.0

        return AgentResult(
            status=final_status,
            agent_name="ExecutionEngine",
            data={
                "plan_id": getattr(plan, "plan_id", ""),
                "objective": getattr(plan, "objective", getattr(plan, "query", "")),
                "step_outputs": step_outputs,
                "execution_trace": execution_trace,
                "total_steps": total_steps,
                "successful_steps": success_count,
            },
            message=msg,
            errors=all_errors,
            warnings=all_warnings,
            confidence=avg_confidence,
            evidence=all_evidence,
            execution_time=total_duration,
            metadata={
                "plan_id": getattr(plan, "plan_id", ""),
                "execution_trace": execution_trace,
            },
        )


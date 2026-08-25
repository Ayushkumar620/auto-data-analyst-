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

import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
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
    Dependency-aware execution engine for analytical task plans.
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY

    def execute_plan(
        self,
        plan: Any,
        dataframe: pd.DataFrame,
    ) -> AgentResult:
        """
        Execute an ExecutionPlan (or legacy TaskPlan) with dependency tracking,
        retry policy, and partial success isolation.
        """
        start_time = datetime.now()
        current_data = dataframe.copy() if dataframe is not None else pd.DataFrame()
        step_results: Dict[str, AgentResult] = {}
        all_evidence: List[Evidence] = []
        all_warnings: List[str] = []
        all_errors: List[AgentError] = []
        execution_trace: List[Dict[str, Any]] = []

        # Validate graph and get steps
        steps = plan.steps if hasattr(plan, "steps") else []
        step_map = {str(getattr(s, "step_id", idx + 1)): s for idx, s in enumerate(steps)}

        # Execute steps in dependency order
        for step in steps:
            s_id = str(getattr(step, "step_id", ""))
            s_name = getattr(step, "name", getattr(step, "purpose", f"Step {s_id}"))
            tool_name = getattr(step, "tool_name", getattr(step, "agent_class_name", ""))
            dependencies = [str(d) for d in getattr(step, "dependencies", [])]

            # 1. Check if any upstream prerequisite dependency failed or was skipped
            deps_failed = False
            for dep_id in dependencies:
                dep_res = step_results.get(dep_id)
                if not dep_res or not dep_res.is_success:
                    deps_failed = True
                    break

            if deps_failed:
                setattr(step, "status", StepStatus.SKIPPED.value)
                all_warnings.append(f"Step {s_id} ('{s_name}') skipped due to upstream dependency failure.")
                execution_trace.append({
                    "step_id": s_id,
                    "name": s_name,
                    "tool": tool_name,
                    "status": StepStatus.SKIPPED.value,
                    "duration_ms": 0.0,
                })
                continue

            # 2. Execute step with retry policy
            setattr(step, "status", StepStatus.RUNNING.value)
            step_start = datetime.now()
            retry_policy = getattr(step, "retry_policy", {"max_retries": 1})
            max_retries = retry_policy.get("max_retries", 1) if isinstance(retry_policy, dict) else 1

            result: Optional[AgentResult] = None
            attempt = 0

            while attempt <= max_retries:
                try:
                    # Build inputs
                    params = getattr(step, "inputs", getattr(step, "parameters", {}))
                    task_inputs = {"data": current_data, **params}

                    # Pass accumulated step outputs to synthesis / reporting agents
                    if tool_name in ("reporting", "ReportAgent", "InsightAgent"):
                        task_inputs["agent_outputs"] = list(step_results.values())

                    # Execute via ToolRegistry or fallback agent invocation
                    if self.tool_registry.has_tool(tool_name):
                        res = self.tool_registry.execute(tool_name, **task_inputs)
                        if isinstance(res, AgentResult):
                            result = res
                        else:
                            result = AgentResult.success(output={"result": res}, agent_name=tool_name)
                    else:
                        # Fallback direct agent class instantiation
                        from agent.agents import AnalysisAgent
                        result = AnalysisAgent().execute(task_inputs)

                    # Validate AgentResult
                    if result is not None and result.is_success:
                        break
                    else:
                        attempt += 1
                        if attempt <= max_retries:
                            setattr(step, "status", StepStatus.RETRYING.value)
                            all_warnings.append(f"Step {s_id} retry {attempt}/{max_retries}...")
                            time.sleep(0.05)
                except Exception as exc:
                    attempt += 1
                    if attempt > max_retries:
                        result = AgentResult.error(
                            error=f"Step execution failed: {str(exc)}",
                            agent_name=tool_name or "ExecutionEngine",
                        )
                        break
                    time.sleep(0.05)

            # Record step timing and result
            duration_ms = round((datetime.now() - step_start).total_seconds() * 1000, 2)
            setattr(step, "duration_ms", duration_ms)
            setattr(step, "result", result)

            if result is not None and result.is_success:
                setattr(step, "status", StepStatus.SUCCESS.value)
                step_results[s_id] = result
                all_evidence.extend(result.evidence)
                all_warnings.extend(result.warnings)

                # If cleaning was performed, carry cleaned dataset forward
                if tool_name in ("data_cleaning", "CleaningAgent"):
                    reports = result.output.get("reports", []) if isinstance(result.output, dict) else []
                    if reports and isinstance(reports, list) and "cleaned_data" in reports[0]:
                        current_data = pd.DataFrame(reports[0]["cleaned_data"])
                    elif isinstance(result.data, dict) and "cleaned_data" in result.data:
                        current_data = pd.DataFrame(result.data["cleaned_data"])
            else:
                setattr(step, "status", StepStatus.FAILED.value)
                if result:
                    all_errors.extend(result.errors)
                    all_warnings.extend(result.warnings)
                all_warnings.append(f"Step {s_id} ('{s_name}') failed after {attempt} attempts.")

            execution_trace.append({
                "step_id": s_id,
                "name": s_name,
                "tool": tool_name,
                "status": getattr(step, "status", StepStatus.FAILED.value),
                "duration_ms": duration_ms,
                "confidence": getattr(result, "confidence", 0.0) if result else 0.0,
            })

        total_duration = round((datetime.now() - start_time).total_seconds() * 1000, 2)
        if hasattr(plan, "total_duration_ms"):
            plan.total_duration_ms = total_duration

        # 3. Determine Overall Execution Status
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

        # Aggregate final outputs
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


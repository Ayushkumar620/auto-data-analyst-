# Planner Agent - routes requests across specialized agents
from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from backend.app.core.semantic import detect_identifiers, SemanticSchemaAgent
from backend.app.core.temporal import TemporalIntelligenceEngine
from backend.app.core.anomalies import AnomalyDetectionEngine
from backend.app.core.relationships import RelationshipDiscoveryEngine
from backend.app.services.dataset_service import DatasetService
from backend.app.cleaning.cleaner import DataCleaner
from backend.app.eda.orchestrator import EDAOrchestrator
from backend.app.insights.engine import InsightEngine
from backend.app.visualization.engine import VisualizationEngine
from backend.app.forecasting.forecaster import Forecaster

@dataclass
class Task:
    id: str
    agent: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    result: Optional[dict[str, Any]] = None

@dataclass
class Plan:
    tasks: List[Task] = field(default_factory=list)
    user_goal: str = ''
    dataset_id: Optional[str] = None

class PlannerAgent:
    def __init__(self) -> None:
        self.temporal = TemporalIntelligenceEngine()
        self.anomaly_engine = AnomalyDetectionEngine()
        self.relationship_engine = RelationshipDiscoveryEngine()
    
    def plan(self, user_goal: str, dataset_id: str, dataset_info: dict[str, Any]) -> Plan:
        tasks = []
        goal_lower = user_goal.casefold().strip()
        tasks.append(Task(id='profile_1', agent='schema', action='detect_semantic_roles', parameters={}, depends_on=[]))
        tasks.append(Task(id='identify_ids_1', agent='schema', action='detect_identifiers', parameters={}, depends_on=['profile_1']))
        
        if any(word in goal_lower for word in ('forecast', 'predict', 'projection', 'next')):
            tasks.extend(self._plan_forecasting(goal_lower, dataset_id, dataset_info))
        elif any(word in goal_lower for word in ('trend', 'increase', 'decrease', 'growth', 'rise', 'fall')):
            tasks.extend(self._plan_trend_analysis(goal_lower, dataset_id, dataset_info))
        elif any(word in goal_lower for word in ('anomaly', 'outlier', 'unusual')):
            tasks.extend(self._plan_anomaly_detection(goal_lower, dataset_id, dataset_info))
        elif any(word in goal_lower for word in ('relationship', 'correlation', 'depends on')):
            tasks.extend(self._plan_relationship_discovery(goal_lower, dataset_id, dataset_info))
        elif any(word in goal_lower for word in ('summary', 'describe', 'what is', 'overview')):
            tasks.extend(self._plan_summary(goal_lower, dataset_id, dataset_info))
        else:
            tasks.extend(self._plan_full_analysis(goal_lower, dataset_id, dataset_info))
        
        return Plan(tasks=tasks, user_goal=user_goal, dataset_id=dataset_id)
    
    def _plan_forecasting(self, goal: str, dataset_id: str, dataset_info: dict[str, Any]) -> List[Task]:
        return [
            Task(id='profile_forecast_1', agent='schema', action='detect_semantic_roles', parameters={}, depends_on=[]),
            Task(id='detect_temporal_1', agent='temporal', action='detect_fields', parameters={}, depends_on=['profile_forecast_1']),
            Task(id='validate_forecast_1', agent='temporal', action='analyze_trends', parameters={}, depends_on=['detect_temporal_1']),
        ]
    
    def _plan_trend_analysis(self, goal: str, dataset_id: str, dataset_info: dict[str, Any]) -> List[Task]:
        return [
            Task(id='profile_trend_1', agent='schema', action='detect_semantic_roles', parameters={}, depends_on=[]),
            Task(id='detect_temporal_trend_1', agent='temporal', action='detect_fields', parameters={}, depends_on=['profile_trend_1']),
            Task(id='analyze_trend_1', agent='temporal', action='analyze_trends', parameters={}, depends_on=['detect_temporal_trend_1']),
        ]
    
    def _plan_anomaly_detection(self, goal: str, dataset_id: str, dataset_info: dict[str, Any]) -> List[Task]:
        return [
            Task(id='profile_anomaly_1', agent='schema', action='detect_semantic_roles', parameters={}, depends_on=[]),
            Task(id='detect_anomalies_1', agent='anomalies', action='detect', parameters={}, depends_on=['profile_anomaly_1']),
        ]
    
    def _plan_relationship_discovery(self, goal: str, dataset_id: str, dataset_info: dict[str, Any]) -> List[Task]:
        return [
            Task(id='profile_relations_1', agent='schema', action='detect_semantic_roles', parameters={}, depends_on=[]),
            Task(id='discover_relationships_1', agent='relationships', action='discover', parameters={}, depends_on=['profile_relations_1']),
        ]
    
    def _plan_summary(self, goal: str, dataset_id: str, dataset_info: dict[str, Any]) -> List[Task]:
        return [
            Task(id='profile_summary_1', agent='schema', action='detect_semantic_roles', parameters={}, depends_on=[]),
            Task(id='generate_summary_1', agent='insights', action='generate', parameters={}, depends_on=['profile_summary_1']),
        ]
    
    def _plan_full_analysis(self, goal: str, dataset_id: str, dataset_info: dict[str, Any]) -> List[Task]:
        return [
            Task(id='profile_1', agent='schema', action='detect_semantic_roles', parameters={}, depends_on=[]),
            Task(id='identify_ids_1', agent='schema', action='detect_identifiers', parameters={}, depends_on=['profile_1']),
            Task(id='detect_temporal_1', agent='temporal', action='detect_fields', parameters={}, depends_on=['profile_1']),
            Task(id='detect_anomalies_1', agent='anomalies', action='detect', parameters={}, depends_on=['profile_1']),
            Task(id='discover_relationships_1', agent='relationships', action='discover', parameters={}, depends_on=['profile_1']),
            Task(id='generate_insights_1', agent='insights', action='generate', parameters={}, depends_on=['profile_1', 'detect_anomalies_1', 'discover_relationships_1']),
            Task(id='visualize_1', agent='visualization', action='recommend', parameters={}, depends_on=['generate_insights_1']),
        ]
    
    def execute_plan(self, plan: Plan, dataframe: Any, dataset_service: DatasetService) -> dict[str, Any]:
        results: Dict[str, Any] = {}
        executed = set()
        def execute_task(task: Task) -> Any:
            if task.id in executed:
                return results[task.id]
            for dep_id in task.depends_on:
                if dep_id not in executed:
                    dep_task = next((t for t in plan.tasks if t.id == dep_id), None)
                    if dep_task:
                        execute_task(dep_task)
            result = self._run_agent_task(task, dataframe, dataset_service)
            results[task.id] = result
            executed.add(task.id)
            return result
        for task in plan.tasks:
            execute_task(task)
        return results
    
    def _run_agent_task(self, task: Task, dataframe: Any, dataset_service: DatasetService) -> dict[str, Any]:
        agent_name = task.agent
        action = task.action
        if agent_name == 'schema':
            return self._run_schema_task(action, dataframe)
        elif agent_name == 'temporal':
            return self._run_temporal_task(action, dataframe)
        elif agent_name == 'anomalies':
            return self._run_anomaly_task(action, dataframe)
        elif agent_name == 'relationships':
            return self._run_relationship_task(action, dataframe)
        elif agent_name == 'insights':
            return self._run_insight_task(action, dataframe)
        elif agent_name == 'visualization':
            return self._run_visualization_task(action, dataframe)
        elif agent_name == 'forecasting':
            return self._run_forecasting_task(action, dataframe)
        return {'status': 'unknown_task'}
    
    def _run_schema_task(self, action: str, dataframe: Any) -> dict[str, Any]:
        if action == 'detect_semantic_roles':
            agent = SemanticSchemaAgent()
            roles = agent.classify(dataframe)
            return {'semantic_roles': roles}
        elif action == 'detect_identifiers':
            identifiers = detect_identifiers(dataframe)
            return {'identifiers': identifiers}
        return {'status': f'schema_action_{action}'}
    
    def _run_temporal_task(self, action: str, dataframe: Any) -> dict[str, Any]:
        if action == 'detect_fields':
            fields = self.temporal.detect_fields(dataframe)
            return {'temporal_fields': fields}
        elif action == 'analyze_trends':
            trends = self.temporal.analyze_trends(dataframe)
            return {'trends': trends}
        return {'status': f'temporal_action_{action}'}
    
    def _run_anomaly_task(self, action: str, dataframe: Any) -> dict[str, Any]:
        if action == 'detect':
            result = self.anomaly_engine.detect(dataframe, method='auto')
            return {'anomalies': result}
        return {'status': f'anomaly_action_{action}'}
    
    def _run_relationship_task(self, action: str, dataframe: Any) -> dict[str, Any]:
        if action == 'discover':
            result = self.relationship_engine.discover(dataframe)
            return {'relationships': result}
        return {'status': f'relationships_action_{action}'}
    
    def _run_insight_task(self, action: str, dataframe: Any) -> dict[str, Any]:
        if action == 'generate':
            engine = InsightEngine()
            result = engine.generate(dataframe)
            return {'insights': result}
        return {'status': f'insights_action_{action}'}
    
    def _run_visualization_task(self, action: str, dataframe: Any) -> dict[str, Any]:
        if action == 'recommend':
            engine = VisualizationEngine()
            recommendations = engine.recommend(dataframe)
            return {'visualization_recommendations': recommendations}
        return {'status': f'visualization_action_{action}'}
    
    def _run_forecasting_task(self, action: str, dataframe: Any) -> dict[str, Any]:
        if action == 'forecast':
            engine = Forecaster()
            try:
                result = engine.forecast(dataframe)
                return {'forecast': result}
            except Exception as exc:
                return {'error': str(exc)}
        return {'status': f'forecasting_action_{action}'}

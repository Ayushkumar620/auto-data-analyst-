"""
Unit and Integration Tests for Automated Continuous Alerting and Live Model Serving.
"""
import pytest
from agent.alert_engine import AlertRuleConfig, EnterpriseAlertEngine
from agent.model_serving import EnterpriseModelServingManager


def test_alert_engine_dispatch():
    engine = EnterpriseAlertEngine()
    rule = AlertRuleConfig(
        name="Test Slack Alert Channel",
        channel_type="slack",
        webhook_url="https://example.com/webhooks/slack-incoming",
        enabled=True,
    )
    engine.save_rule(rule)

    dispatches = engine.dispatch_alert(
        title="Critical Drift Incident",
        message="Feature 'transaction_amount' KS p-value = 0.0002 (< 0.01 threshold)",
        severity="CRITICAL",
    )
    assert len(dispatches) >= 1
    assert dispatches[0].severity == "CRITICAL"
    assert dispatches[0].delivery_status in ("DELIVERED", "MOCKED_SUCCESS")

    history = engine.list_history()
    assert len(history) >= 1
    assert history[0].title == "Critical Drift Incident"


def test_model_serving_deployment_and_prediction():
    manager = EnterpriseModelServingManager()
    dep = manager.deploy_model("mod_test_123", endpoint_name="churn_predictor_v1")
    assert dep.status == "ACTIVE"
    assert dep.model_id == "mod_test_123"

    res = manager.predict_endpoint(dep.deployment_id, [{"feature_1": 10.0, "feature_2": 20.0}])
    assert res.deployment_id == dep.deployment_id
    assert len(res.predictions) == 1
    assert res.latency_ms >= 0.0

    # Test undeploy
    deleted = manager.undeploy(dep.deployment_id)
    assert deleted is True
    assert manager.get_deployment(dep.deployment_id) is None

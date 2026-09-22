"""Tests for drift monitoring, audit logging, HITL, and policy engine."""

import json
import numpy as np
import pytest

from responsible_ai_toolkit.drift.monitor import DriftMonitor, DriftReport
from responsible_ai_toolkit.audit.logger import AuditLogger, EventType
from responsible_ai_toolkit.hitl.orchestrator import HITLOrchestrator, CaseStatus
from responsible_ai_toolkit.policy.engine import PolicyEngine, Policy


# ---------------------------------------------------------------------------
# DriftMonitor tests
# ---------------------------------------------------------------------------

class TestDriftMonitor:

    def test_no_drift(self):
        rng = np.random.RandomState(42)
        reference = rng.normal(0.5, 0.1, 1000)
        production = rng.normal(0.5, 0.1, 1000)
        monitor = DriftMonitor(reference)
        report = monitor.evaluate(production)
        assert isinstance(report, DriftReport)
        assert report.psi < 0.10
        assert "No significant drift" in report.psi_interpretation

    def test_significant_drift(self):
        rng = np.random.RandomState(42)
        reference = rng.normal(0.5, 0.1, 1000)
        production = rng.normal(0.8, 0.2, 1000)  # Shifted distribution
        monitor = DriftMonitor(reference)
        report = monitor.evaluate(production)
        assert report.psi > 0.25
        assert "Significant drift" in report.psi_interpretation

    def test_psi_non_negative(self):
        rng = np.random.RandomState(42)
        reference = rng.uniform(0, 1, 500)
        production = rng.uniform(0, 1, 500)
        monitor = DriftMonitor(reference)
        report = monitor.evaluate(production)
        assert report.psi >= 0

    def test_kl_divergence_non_negative(self):
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 500)
        production = rng.normal(0.5, 1, 500)
        monitor = DriftMonitor(reference)
        report = monitor.evaluate(production)
        assert report.kl_divergence >= 0

    def test_wasserstein_non_negative(self):
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 500)
        production = rng.normal(0, 1, 500)
        monitor = DriftMonitor(reference)
        report = monitor.evaluate(production)
        assert report.wasserstein_distance >= 0

    def test_bin_details(self):
        rng = np.random.RandomState(42)
        reference = rng.uniform(0, 1, 500)
        production = rng.uniform(0, 1, 500)
        monitor = DriftMonitor(reference, n_bins=5)
        report = monitor.evaluate(production)
        assert len(report.bin_details) == 5

    def test_report_summary(self):
        rng = np.random.RandomState(42)
        reference = rng.normal(0, 1, 500)
        monitor = DriftMonitor(reference)
        report = monitor.evaluate(rng.normal(0, 1, 500))
        summary = report.summary()
        assert "MODEL DRIFT REPORT" in summary
        assert "PSI:" in summary

    def test_insufficient_reference_data(self):
        with pytest.raises(ValueError, match="at least"):
            DriftMonitor(np.array([0.1, 0.2]), n_bins=10)


# ---------------------------------------------------------------------------
# AuditLogger tests
# ---------------------------------------------------------------------------

class TestAuditLogger:

    def test_log_prediction(self):
        logger = AuditLogger(system_id="test-system")
        entry = logger.log_prediction(
            model_id="model-v1",
            input_data={"feature": 42},
            output={"score": 0.85},
        )
        assert entry.system_id == "test-system"
        assert entry.event_type == "prediction"
        assert logger.size == 1

    def test_hash_chain_integrity(self):
        logger = AuditLogger(system_id="test-system")
        for i in range(10):
            logger.log_prediction(
                model_id="model-v1",
                input_data={"i": i},
                output={"score": 0.5},
            )
        assert logger.verify_chain()

    def test_tamper_detection(self):
        logger = AuditLogger(system_id="test-system")
        logger.log_prediction("m1", {"a": 1}, {"b": 2})
        logger.log_prediction("m1", {"a": 3}, {"b": 4})
        # Tamper with the first entry
        logger._entries[0].entry_hash = "tampered_hash"
        assert not logger.verify_chain()

    def test_genesis_hash(self):
        logger = AuditLogger(system_id="test")
        entry = logger.log_prediction("m1", {}, {})
        assert entry.previous_hash == AuditLogger.GENESIS_HASH

    def test_chain_linkage(self):
        logger = AuditLogger(system_id="test")
        e1 = logger.log_prediction("m1", {}, {})
        e2 = logger.log_prediction("m1", {}, {})
        assert e2.previous_hash == e1.entry_hash

    def test_log_human_review(self):
        logger = AuditLogger(system_id="test")
        entry = logger.log_human_review(
            reviewer_id="analyst_1",
            decision="approve",
            reasoning="Consistent with policy",
            original_prediction={"score": 0.87},
        )
        assert entry.event_type == "human_review"

    def test_log_fairness_evaluation(self):
        logger = AuditLogger(system_id="test")
        entry = logger.log_fairness_evaluation(
            evaluation_id="eval-001",
            metrics={"demographic_parity": {"A": 0.5, "B": 0.45}},
            violations=[],
        )
        assert entry.event_type == "fairness_evaluation"

    def test_export_json(self):
        logger = AuditLogger(system_id="test")
        logger.log_prediction("m1", {"x": 1}, {"y": 2})
        exported = logger.export_json()
        data = json.loads(exported)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_evidence_package(self):
        logger = AuditLogger(system_id="test")
        logger.log_prediction("m1", {}, {})
        logger.log_policy_check("p1", "pass", {})
        package = logger.export_evidence_package()
        assert package["chain_integrity"] is True
        assert package["total_entries"] == 2

    def test_filter_by_type(self):
        logger = AuditLogger(system_id="test")
        logger.log_prediction("m1", {}, {})
        logger.log_prediction("m1", {}, {})
        logger.log_human_review("r1", "approve", "ok", {})
        preds = logger.get_entries_by_type(EventType.PREDICTION)
        assert len(preds) == 2


# ---------------------------------------------------------------------------
# HITLOrchestrator tests
# ---------------------------------------------------------------------------

class TestHITLOrchestrator:

    def test_submit_case(self):
        hitl = HITLOrchestrator()
        hitl.add_reviewer("analyst_1", roles=["lending"])
        case = hitl.submit_for_review(
            case_id="LOAN-001",
            category="lending",
            ai_decision={"approved": True},
            reason="Low confidence",
            priority="high",
        )
        assert case.status == CaseStatus.ASSIGNED
        assert case.assigned_to == "analyst_1"

    def test_no_reviewer_stays_pending(self):
        hitl = HITLOrchestrator()
        case = hitl.submit_for_review(
            case_id="LOAN-001",
            category="lending",
            ai_decision={},
            reason="test",
        )
        assert case.status == CaseStatus.PENDING

    def test_record_decision(self):
        hitl = HITLOrchestrator()
        hitl.add_reviewer("analyst_1", roles=["lending"])
        case = hitl.submit_for_review(
            case_id="LOAN-001",
            category="lending",
            ai_decision={"approved": True, "score": 0.55},
            reason="Low confidence",
        )
        result = hitl.record_decision(
            internal_id=case.internal_id,
            reviewer_id="analyst_1",
            decision="override",
            reasoning="Applicant has strong collateral not captured by model",
        )
        assert result.status == CaseStatus.COMPLETED
        assert result.decision.value == "override"

    def test_should_review_low_confidence(self):
        hitl = HITLOrchestrator(
            confidence_threshold=0.70,
            active_learning_rate=0.0,
        )
        assert hitl.should_review(0.50) is True
        assert hitl.should_review(0.70) is False
        assert hitl.should_review(0.90) is False

    def test_should_review_when_sampling_is_enabled(self):
        hitl = HITLOrchestrator(
            confidence_threshold=0.70,
            active_learning_rate=1.0,
        )
        assert hitl.should_review(0.90) is True

    def test_sla_compliance_rate(self):
        hitl = HITLOrchestrator()
        hitl.add_reviewer("analyst_1")
        case = hitl.submit_for_review(
            case_id="C-001", category="general",
            ai_decision={}, reason="test",
        )
        hitl.record_decision(case.internal_id, "analyst_1", "approve", "ok")
        assert hitl.sla_compliance_rate() == 1.0

    def test_get_stats(self):
        hitl = HITLOrchestrator()
        hitl.add_reviewer("analyst_1")
        stats = hitl.get_stats()
        assert "total_cases" in stats
        assert "sla_compliance_rate" in stats
        assert stats["total_reviewers"] == 1

    def test_role_based_routing(self):
        hitl = HITLOrchestrator()
        hitl.add_reviewer("lending_analyst", roles=["lending"])
        hitl.add_reviewer("fraud_analyst", roles=["fraud"])
        case = hitl.submit_for_review(
            case_id="F-001", category="fraud",
            ai_decision={}, reason="Suspicious pattern",
        )
        assert case.assigned_to == "fraud_analyst"


# ---------------------------------------------------------------------------
# PolicyEngine tests
# ---------------------------------------------------------------------------

class TestPolicyEngine:

    def test_add_and_evaluate_policy(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            policy_id="test-001",
            name="Score Threshold",
            description="Score must be >= 0.5",
            rule=lambda ctx: ctx.get("score", 0) >= 0.5,
        ))
        result = engine.evaluate({"score": 0.7})
        assert result.all_passed

    def test_violation_detected(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            policy_id="test-001",
            name="Score Threshold",
            description="Score must be >= 0.5",
            rule=lambda ctx: ctx.get("score", 0) >= 0.5,
            severity="critical",
        ))
        result = engine.evaluate({"score": 0.3})
        assert not result.all_passed
        assert len(result.critical_violations) == 1

    def test_factory_fairness_policy(self):
        engine = PolicyEngine()
        policy = PolicyEngine.fairness_threshold_policy(
            "fair-001", "dp_ratio", threshold=0.80
        )
        engine.add_policy(policy)
        assert engine.evaluate({"dp_ratio": 0.85}).all_passed
        assert not engine.evaluate({"dp_ratio": 0.70}).all_passed

    def test_factory_confidence_policy(self):
        engine = PolicyEngine()
        policy = PolicyEngine.confidence_threshold_policy("conf-001", threshold=0.60)
        engine.add_policy(policy)
        assert engine.evaluate({"confidence": 0.75}).all_passed
        assert not engine.evaluate({"confidence": 0.45}).all_passed

    def test_factory_drift_policy(self):
        engine = PolicyEngine()
        policy = PolicyEngine.drift_threshold_policy("drift-001", psi_threshold=0.25)
        engine.add_policy(policy)
        assert engine.evaluate({"psi": 0.10}).all_passed
        assert not engine.evaluate({"psi": 0.30}).all_passed

    def test_multiple_policies(self):
        engine = PolicyEngine()
        engine.add_policy(Policy("p1", "A", "desc", lambda c: c.get("a", 0) > 0))
        engine.add_policy(Policy("p2", "B", "desc", lambda c: c.get("b", 0) > 0))
        result = engine.evaluate({"a": 1, "b": 0})
        assert result.pass_rate == 0.5

    def test_framework_filtering(self):
        engine = PolicyEngine()
        engine.add_policy(Policy("p1", "A", "d", lambda c: True, frameworks=["ECOA"]))
        engine.add_policy(Policy("p2", "B", "d", lambda c: True, frameworks=["NIST"]))
        result = engine.evaluate({}, frameworks=["ECOA"])
        assert len(result.results) == 1

    def test_report_summary(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            "p1", "Failing Policy", "Always fails",
            lambda c: False, severity="critical"
        ))
        result = engine.evaluate({})
        summary = result.summary()
        assert "POLICY EVALUATION REPORT" in summary
        assert "Failing Policy" in summary

    def test_error_handling(self):
        engine = PolicyEngine()
        engine.add_policy(Policy(
            "p1", "Broken", "Will error",
            lambda c: c["nonexistent_key"],  # Will raise KeyError
        ))
        result = engine.evaluate({})
        assert not result.all_passed
        assert result.results[0].error is not None

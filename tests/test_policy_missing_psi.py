from responsible_ai_toolkit.policy import PolicyEngine


def test_missing_psi_does_not_pass():
    engine = PolicyEngine()
    engine.add_policy(
        engine.drift_threshold_policy("DRIFT-001")
    )

    report = engine.evaluate({})

    assert len(report.results) == 1
    result = report.results[0]
    assert result.passed is False
    assert result.error is not None
    assert "psi" in result.error
    assert report.all_passed is False


def test_valid_psi_below_threshold_passes():
    engine = PolicyEngine()
    engine.add_policy(
        engine.drift_threshold_policy("DRIFT-001")
    )

    report = engine.evaluate({"psi": 0.10})

    assert report.all_passed is True
    assert report.results[0].error is None


def test_psi_at_threshold_does_not_pass():
    engine = PolicyEngine()
    engine.add_policy(
        engine.drift_threshold_policy("DRIFT-001")
    )

    report = engine.evaluate({"psi": 0.25})

    assert report.all_passed is False
    assert report.results[0].error is None

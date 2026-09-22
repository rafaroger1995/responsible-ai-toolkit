import pytest

from responsible_ai_toolkit.audit import AuditLogger


def _logger_with_entries():
    logger = AuditLogger(system_id="lending-model-v2")
    logger.log_prediction(
        model_id="credit-score-v2",
        input_data={"income": 75000},
        output={"approved": True},
    )
    logger.log_policy_check(policy_id="P-1", result="passed", details={})
    logger.log_drift_alert(
        metric_name="psi",
        value=0.1,
        threshold=0.25,
        interpretation="stable",
    )
    return logger


def test_untampered_chain_verifies():
    assert _logger_with_entries().verify_chain()


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize(
    "field_name, new_value",
    [
        ("system_id", "other-system"),
        ("timestamp_iso", "2000-01-01T00:00:00Z"),
        ("timestamp", 0.0),
        ("entry_id", "forged-id"),
        ("sequence_number", 99),
        ("event_type", "override"),
    ],
)
def test_changing_an_entry_field_is_detected(index, field_name, new_value):
    logger = _logger_with_entries()
    setattr(logger.entries[index], field_name, new_value)
    assert not logger.verify_chain()


def test_changing_payload_is_detected():
    logger = _logger_with_entries()
    logger.entries[1].payload["result"] = "failed"
    assert not logger.verify_chain()


def test_removing_a_middle_entry_is_detected():
    logger = _logger_with_entries()
    del logger._entries[1]
    assert not logger.verify_chain()


def test_reordering_entries_is_detected():
    logger = _logger_with_entries()
    logger._entries[0], logger._entries[1] = logger._entries[1], logger._entries[0]
    assert not logger.verify_chain()


def test_caller_changing_its_own_data_does_not_alter_log():
    logger = AuditLogger(system_id="lending-model-v2")
    input_data = {"income": 75000}
    logger.log_prediction(
        model_id="credit-score-v2",
        input_data=input_data,
        output={"approved": True},
    )

    input_data["income"] = 1

    assert logger.verify_chain()
    assert logger.entries[0].payload["input"]["income"] == 75000

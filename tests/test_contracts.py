import math

import pytest

from troubleshooter.contracts import normalize_result
from troubleshooter.stub import StubState, stub_result


def raw():
    return stub_result("success", "TP_DEMO", "req", StubState())


@pytest.mark.parametrize(
    "changes",
    [
        {"contact_verified": False},
        {"contact": "unknown"},
        {"move_completed": False},
        {"position_ok": False},
        {"board_unmoved": False},
        {"calibration_valid": False},
        {"retraction_status": "failed"},
        {"stop_confirmed": False},
        {"reading": math.nan},
        {"reading": math.inf},
        {"units": "mV"},
        {"mode": "guess"},
        {"mode": None},
        {"request_id": "another"},
        {"requested_target": "another"},
        {"registration_id": ""},
        {"test_point": ""},
        {"contact_verified": "true"},
    ],
)
def test_inconsistent_success_is_never_evidence(changes):
    result = normalize_result(raw() | changes, "req", "TP_DEMO")
    assert result.status == "inconclusive"
    assert result.failure_reason == "protocol_violation"
    assert result.reading is None
    assert result.execution_uncertain
    assert "NaN" not in result.model_dump_json()


def test_zero_with_unverified_contact_is_not_a_reading():
    result = normalize_result(
        raw()
        | {
            "status": "inconclusive",
            "reading": 0.0,
            "contact_verified": False,
            "contact": "not_verified",
            "failure_reason": "contact_unverified",
        },
        "req",
        "TP_DEMO",
    )
    assert result.reading is None
    assert result.failure_reason == "protocol_violation"


def test_valid_zero_is_preserved_and_recovery_log_is_not_discarded():
    result = normalize_result(
        raw()
        | {
            "reading": 0.0,
            "recovery_log": ['{"source":"deterministic"}', "raw log entry"],
        },
        "req",
        "TP_DEMO",
    )
    assert result.status == "ok"
    assert result.reading == 0.0
    assert result.recovery_log == [{"source": "deterministic"}, "raw log entry"]

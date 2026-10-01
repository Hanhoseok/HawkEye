"""사건 규칙 (docs/design.md §4).

한 손님에게서 경보가 여러 번 오므로, 들어온 경보가 사건 상태를 어떻게 바꾸고
폰 알림을 띄울지 정하는 규칙을 시험한다.
"""

import pytest

from hawkeye_server.cases import Current, decide


def test_new_warning_opens_active_case_and_notifies():
    d = decide(None, "WARNING")
    assert (d.level, d.state, d.notify) == ("WARNING", "active", True)


def test_new_clear_is_a_silent_pass():
    d = decide(None, "CLEAR")
    assert (d.level, d.state, d.notify) == ("CLEAR", "pass", False)


def test_warning_escalates_to_high_risk_with_notification():
    d = decide(Current("WARNING", "active"), "HIGH_RISK")
    assert (d.level, d.state, d.notify) == ("HIGH_RISK", "active", True)


def test_repeated_same_level_does_not_notify_again():
    d = decide(Current("WARNING", "active"), "WARNING")
    assert (d.level, d.state, d.notify) == ("WARNING", "active", False)


def test_warning_then_clear_is_auto_cleared_silently():
    d = decide(Current("WARNING", "active"), "CLEAR")
    assert (d.level, d.state, d.notify) == ("CLEAR", "auto_cleared", False)


def test_confirmed_high_risk_is_never_lowered_by_clear():
    d = decide(Current("HIGH_RISK", "active"), "CLEAR")
    assert (d.level, d.state, d.notify) == ("HIGH_RISK", "active", False)


def test_lower_alarm_does_not_lower_level():
    d = decide(Current("HIGH_RISK", "active"), "REVIEW")
    assert (d.level, d.state, d.notify) == ("HIGH_RISK", "active", False)


def test_review_escalates_to_high_risk():
    d = decide(Current("REVIEW", "active"), "HIGH_RISK")
    assert (d.level, d.state, d.notify) == ("HIGH_RISK", "active", True)


def test_escalation_reopens_a_resolved_case():
    d = decide(Current("WARNING", "resolved"), "HIGH_RISK")
    assert (d.level, d.state, d.notify, d.reopened) == ("HIGH_RISK", "active", True, True)


def test_same_level_keeps_resolved_case_closed():
    d = decide(Current("HIGH_RISK", "resolved"), "HIGH_RISK")
    assert (d.level, d.state, d.notify, d.reopened) == ("HIGH_RISK", "resolved", False, False)


def test_reapproach_after_auto_clear_alarms_again():
    d = decide(Current("CLEAR", "auto_cleared"), "WARNING")
    assert (d.level, d.state, d.notify) == ("WARNING", "active", True)


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError):
        decide(None, "PANIC")


def test_clear_does_not_override_manager_resolution():
    d = decide(Current("WARNING", "resolved"), "CLEAR")
    assert (d.level, d.state, d.notify) == ("WARNING", "resolved", False)

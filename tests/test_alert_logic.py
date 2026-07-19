"""
tests/test_alert_logic.py
----------------------------
Tests decide_alert() -- the rule that fuses the RUL estimate and anomaly
flag into one actionable alert level. This is pure decision logic (no
model inference inside it), so these tests pin down the POLICY, not the
models: given known inputs, does the alert level come out right, and does
it stay right if someone edits the thresholds later without meaning to
change the boundary behavior?

Uses the real pipeline fixture (see conftest.py) since decide_alert is a
bound method, but none of these tests trigger actual model inference --
they call decide_alert directly with hand-picked (predicted_rul,
is_anomaly) inputs.
"""

from predict import RUL_URGENT_THRESHOLD, RUL_WATCH_THRESHOLD


def test_low_rul_is_urgent_regardless_of_anomaly(pipeline):
    """Below the urgent threshold, alert must be URGENT even with no
    anomaly flag -- RUL alone is sufficient to trigger the most severe level."""
    level, _ = pipeline.decide_alert(predicted_rul=5.0, is_anomaly=False)
    assert level == "URGENT"

    level, _ = pipeline.decide_alert(predicted_rul=5.0, is_anomaly=True)
    assert level == "URGENT"


def test_anomaly_plus_declining_rul_is_warning(pipeline):
    """Anomaly detected AND RUL in the watch window (but not yet urgent)
    should escalate to WARNING -- this is the actual signal-fusion case,
    the reason the pipeline runs two models instead of one."""
    rul_in_watch_window = (RUL_URGENT_THRESHOLD + RUL_WATCH_THRESHOLD) / 2
    level, reason = pipeline.decide_alert(predicted_rul=rul_in_watch_window, is_anomaly=True)
    assert level == "WARNING"
    assert "anomalous" in reason.lower() or "anomaly" in reason.lower()


def test_anomaly_alone_with_healthy_rul_is_watch_not_urgent(pipeline):
    """An anomaly with an otherwise healthy RUL estimate should prompt
    monitoring (WATCH), not a false URGENT -- this is exactly the
    'failure mode the RUL model wasn't trained to recognize' case the
    autoencoder exists to catch, and it shouldn't be over- or under-stated."""
    healthy_rul = RUL_WATCH_THRESHOLD + 50
    level, _ = pipeline.decide_alert(predicted_rul=healthy_rul, is_anomaly=True)
    assert level == "WATCH"


def test_no_anomaly_high_rul_is_healthy(pipeline):
    level, _ = pipeline.decide_alert(predicted_rul=200.0, is_anomaly=False)
    assert level == "HEALTHY"


def test_no_anomaly_watch_window_rul_is_watch(pipeline):
    rul_in_watch_window = (RUL_URGENT_THRESHOLD + RUL_WATCH_THRESHOLD) / 2
    level, _ = pipeline.decide_alert(predicted_rul=rul_in_watch_window, is_anomaly=False)
    assert level == "WATCH"


def test_threshold_ordering_is_sane():
    """Regression pin: RUL_URGENT_THRESHOLD must stay strictly less than
    RUL_WATCH_THRESHOLD for decide_alert's branching order to make sense.
    If someone edits these constants later and flips the ordering, this
    test catches it immediately instead of silently producing wrong alerts."""
    assert RUL_URGENT_THRESHOLD < RUL_WATCH_THRESHOLD

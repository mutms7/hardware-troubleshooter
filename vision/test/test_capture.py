import numpy as np
from hardware_troubleshooter_vision.capture import Frame, JointHistory, JointSample, tag_wrist_frame
from hardware_troubleshooter_vision.config import CaptureConfig

CONFIG = CaptureConfig(max_joint_skew_s=0.01, settle_velocity_rad_s=0.01, settle_duration_s=0.2)
IMAGE = np.zeros((4, 4), np.uint8)


def history(velocity_at=None, start=0.0, end=1.0, step=0.005):
    h = JointHistory()
    for stamp in np.arange(start, end, step):
        v = 0.5 if velocity_at is not None and abs(stamp - velocity_at) < 0.01 else 0.0
        h.add(JointSample(float(stamp), (0.0,) * 7, (v,) * 7))
    return h


def test_settled_frame_is_tagged():
    tagged, reason = tag_wrist_frame(Frame("left", 0.9, IMAGE), "left", history(), CONFIG)
    assert reason is None
    assert tagged.arm == "left" and abs(tagged.joints.stamp - 0.9) <= 0.005


def test_motion_in_the_settle_window_rejects():
    tagged, reason = tag_wrist_frame(
        Frame("left", 0.9, IMAGE), "left", history(velocity_at=0.8), CONFIG
    )
    assert tagged is None and reason.startswith("arm_not_settled:joint speed")


def test_motion_before_the_window_is_fine():
    _, reason = tag_wrist_frame(Frame("left", 0.9, IMAGE), "left", history(velocity_at=0.5), CONFIG)
    assert reason is None


def test_stale_joint_state_rejects():
    tagged, reason = tag_wrist_frame(Frame("left", 1.5, IMAGE), "left", history(), CONFIG)
    assert tagged is None and reason.startswith("joint_state_skew")


def test_short_history_rejects():
    tagged, reason = tag_wrist_frame(Frame("left", 0.95, IMAGE), "left", history(start=0.9), CONFIG)
    assert tagged is None and "doesn't cover the settle window" in reason


def test_empty_history_and_unset_config():
    assert tag_wrist_frame(Frame("l", 1, IMAGE), "l", JointHistory(), CONFIG) == (
        None,
        "no_joint_state",
    )
    tagged, reason = tag_wrist_frame(Frame("l", 1, IMAGE), "l", history(), CaptureConfig())
    assert tagged is None and reason.startswith("unconfigured:capture.")


def test_history_ignores_out_of_order_and_ages_out():
    h = JointHistory(max_age_s=1.0)
    h.add(JointSample(1.0, (0,), (0,)))
    h.add(JointSample(0.5, (0,), (0,)))
    assert h.latest.stamp == 1.0
    h.add(JointSample(2.5, (0,), (0,)))
    assert h.window(0, 3) == [h.latest]
    assert h.nearest(2.0).stamp == 2.5


def test_missing_velocity_nan_and_feedback_gap_reject():
    for velocities in ((), (float("nan"),) * 7):
        h = JointHistory()
        for stamp in np.arange(0, 1, 0.005):
            h.add(JointSample(float(stamp), (0.0,) * 7, velocities))
        _, reason = tag_wrist_frame(Frame("left", 0.9, IMAGE), "left", h, CONFIG)
        assert reason.startswith("invalid_joint_state")
    h = JointHistory()
    for stamp in [0.7, 0.705, 0.89, 0.895, 0.9]:
        h.add(JointSample(stamp, (0.0,) * 7, (0.0,) * 7))
    _, reason = tag_wrist_frame(Frame("left", 0.9, IMAGE), "left", h, CONFIG)
    assert reason == "arm_not_settled:gap in joint feedback"

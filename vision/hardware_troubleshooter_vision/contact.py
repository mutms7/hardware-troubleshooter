"""Touchdown from joint torque: a sustained step away from the pre-descent baseline.

The Maker Arm reports measured torque per joint. When the probe meets the board the
reaction shows up on the pitch joints. That says the tip met *something*; the wrist
alignment before descent says where, and only the AD3 can say the contact is
electrical. None of these alone is contact verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capture import JointSample
from .config import ContactConfig


@dataclass(frozen=True)
class Touchdown:
    stamp: float  # first sample of the persistent step
    joint: int  # 1-based SDK joint that crossed first
    delta_nm: float


class TouchdownDetector:
    """Feed samples during one descent; `update` returns the touchdown once confirmed."""

    def __init__(self, config: ContactConfig):
        self.config = config
        self.baseline: np.ndarray | None = None
        self._since: dict[int, float] = {}
        self.touchdown: Touchdown | None = None

    def start(self, history: list[JointSample], descent_stamp: float) -> str | None:
        """Average watched torques over baseline_s before the descent. None or a reason."""
        missing = self.config.missing()
        if missing:
            return "; ".join(f"unconfigured:{key}" for key in missing)
        window = [
            s for s in history if descent_stamp - self.config.baseline_s <= s.stamp <= descent_stamp
        ]
        if not window:
            return "no_torque_baseline:no samples in the baseline window"
        if any(len(s.torques) < max(self.config.joints) for s in window):
            return "no_torque_baseline:samples carry no torque for the watched joints"
        columns = [j - 1 for j in self.config.joints]
        self.baseline = np.array([[s.torques[c] for c in columns] for s in window]).mean(axis=0)
        self._since, self.touchdown = {}, None
        return None

    def update(self, sample: JointSample) -> Touchdown | None:
        if self.baseline is None:
            raise RuntimeError("start() the descent before feeding samples")
        if self.touchdown is not None:
            return self.touchdown
        for joint, base in zip(self.config.joints, self.baseline):
            delta = sample.torques[joint - 1] - base
            if abs(delta) >= self.config.torque_step_nm:
                since = self._since.setdefault(joint, sample.stamp)
                if sample.stamp - since >= self.config.persistence_s:
                    self.touchdown = Touchdown(since, joint, float(delta))
                    return self.touchdown
            else:
                self._since.pop(joint, None)
        return None

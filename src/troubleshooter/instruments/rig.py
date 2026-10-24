"""Rig wiring and capture settings, loaded from the operator's JSON file (TH_RIG_CONFIG).

Nothing here has a default wiring: an unset file means the instrument tools refuse to run.
The model only ever picks a profile or logic type by name, never raw instrument settings.
"""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NAME_PATTERN = re.compile(r"[A-Za-z0-9_\-]{1,32}\Z")
Protocol = Literal["uart", "pwm", "i2c", "spi"]
# Signal lines each protocol needs. A logic probe carries one line, so two-line protocols use
# both arms' logic probes. SPI gets SCK plus one data line: no chip select, so it's partial.
ROLES: dict[str, tuple[str, ...]] = {
    "uart": ("rx",),
    "pwm": ("signal",),
    "i2c": ("scl", "sda"),
    "spi": ("sck", "data"),
}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScopeHead(Strict):
    channel: Literal[1, 2]  # AD3 scope input; tie its negative input to the board ground too


class LogicHead(Strict):
    dio: int = Field(ge=0, le=15)  # AD3 DIO pin: 3.3 V LVCMOS input, 5 V tolerant


class ArmConfig(Strict):
    url: str  # this arm's sim_server.py
    scope: ScopeHead
    logic: LogicHead
    # The third head is the multimeter probe. The DMM is a separate device, not the AD3,
    # and isn't driven by this package yet.


class ScopeProfile(Strict):
    description: str = ""
    range_v: float = Field(gt=0, le=50)  # full span, peak to peak
    offset_v: float = Field(default=0.0, ge=-25, le=25)
    sample_rate_hz: float = Field(gt=0, le=125e6)
    samples: int = Field(ge=64, le=32768)
    trigger_level_v: float | None = None  # None captures immediately, without a trigger
    trigger_slope: Literal["rising", "falling"] = "rising"
    auto_timeout_s: float = Field(default=1.0, gt=0, le=10)
    require_trigger: bool = False  # an auto-triggered capture is then inconclusive


class LogicType(Strict):
    description: str = ""
    protocol: Protocol
    sample_rate_hz: float = Field(gt=0, le=125e6)
    samples: int = Field(ge=64, le=65536)
    auto_timeout_s: float = Field(default=1.0, gt=0, le=10)
    baud: int | None = Field(default=None, gt=0)
    data_bits: int = Field(default=8, ge=5, le=9)
    parity: Literal["none", "even", "odd"] = "none"
    stop_bits: Literal[1, 2] = 1
    cpol: Literal[0, 1] = 0
    cpha: Literal[0, 1] = 0
    bit_order: Literal["msb", "lsb"] = "msb"

    @model_validator(mode="after")
    def check_protocol(self):
        if self.protocol == "uart":
            if self.baud is None:
                raise ValueError("uart logic types need baud")
            if self.sample_rate_hz < 4 * self.baud:
                raise ValueError("uart sample_rate_hz must be at least 4x baud")
        return self


class RigConfig(Strict):
    arms: dict[str, ArmConfig] = Field(min_length=1)
    scope_profiles: dict[str, ScopeProfile]
    logic_types: dict[str, LogicType]

    @model_validator(mode="after")
    def check_wiring(self):
        for group in (self.arms, self.scope_profiles, self.logic_types):
            for name in group:
                if not NAME_PATTERN.fullmatch(name):
                    raise ValueError(f"invalid name {name!r}")
        channels = [arm.scope.channel for arm in self.arms.values()]
        pins = [arm.logic.dio for arm in self.arms.values()]
        if len(set(channels)) != len(channels) or len(set(pins)) != len(pins):
            raise ValueError("each arm needs its own scope channel and DIO pin")
        return self


def load_rig(path: str | Path) -> RigConfig:
    return RigConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

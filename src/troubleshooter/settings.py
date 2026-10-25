import os
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .instruments.fake import FakeScenario


def _local_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    return Path(base or Path.home() / ".local" / "share") / "hardware-troubleshooter"


class Settings(BaseSettings):
    # Ignore the old .env by design. No model-provider credentials are needed.
    model_config = SettingsConfigDict(env_prefix="TH_", extra="ignore")
    # No hardware backend exists yet; "none" fails closed instead of simulating.
    backend: Literal["none", "stub"] = "none"
    availability_timeout_s: float = Field(default=1.0, gt=0, le=30)
    outer_timeout_s: float | None = Field(default=None, gt=0, le=600)
    cancel_grace_s: float | None = Field(default=None, gt=0, le=30)
    journal: str = "data/mcp_operations.sqlite3"
    stub_scenario: Literal[
        "success",
        "contact_failed",
        "zero_unverified",
        "board_moved",
        "stage_timeout",
        "slow",
        "stale_calibration",
        "unset_config",
        "malformed",
        "bad_mode",
        "unavailable",
        "cancel_unconfirmed",
        "lost_connection",
        "bad_position",
    ] = "success"
    stub_step_s: float = Field(default=0.03, gt=0, le=1)
    # AD3 instrument tools. Unset rig_config disables them; "fake" is never a fallback.
    rig_config: str | None = None
    instruments: Literal["ad3", "fake"] = "ad3"
    ad3_serial: str | None = None
    fake_scenario: FakeScenario = "success"
    arm_timeout_s: float = Field(default=5.0, gt=0, le=30)
    captures_dir: str = "data/captures"
    # Multimeter knob tools (arm/bench run.py). Unset disables them.
    dmm_knob_config: str | None = None
    # Design and datasheet stores. Kept off synced folders such as OneDrive by default,
    # since a sync client can corrupt an SQLite file it copies mid-write.
    design_dir: str = Field(default_factory=lambda: str(_local_data_dir()))
    # Moss mirrors the datasheet index when both are set. Its own variable names work too.
    moss_project_id: str | None = Field(
        default=None, validation_alias=AliasChoices("TH_MOSS_PROJECT_ID", "MOSS_PROJECT_ID")
    )
    moss_project_key: str | None = Field(
        default=None, validation_alias=AliasChoices("TH_MOSS_PROJECT_KEY", "MOSS_PROJECT_KEY")
    )

    @property
    def effective_outer_timeout(self) -> float | None:
        # Software simulation limits, not hardware acceptance limits.
        return self.outer_timeout_s or (3.0 if self.backend == "stub" else None)

    @property
    def effective_cancel_grace(self) -> float | None:
        return self.cancel_grace_s or (0.5 if self.backend == "stub" else None)

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .instruments.fake import FakeScenario


class Settings(BaseSettings):
    # Ignore the old .env by design. No model-provider credentials are needed.
    model_config = SettingsConfigDict(env_prefix="TH_", extra="ignore")
    backend: Literal["ros", "stub"] = "ros"
    action_name: str = "/measure"
    registry_service: str = "/registry/list"
    status_service: str = "/rig/status"
    availability_timeout_s: float = Field(default=1.0, gt=0, le=30)
    service_timeout_s: float = Field(default=2.0, gt=0, le=30)
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

    @property
    def effective_outer_timeout(self) -> float | None:
        # Software simulation limits, not hardware acceptance limits.
        return self.outer_timeout_s or (3.0 if self.backend == "stub" else None)

    @property
    def effective_cancel_grace(self) -> float | None:
        return self.cancel_grace_s or (0.5 if self.backend == "stub" else None)

"""Write one arm's URDF and Maker SDK profile.

python -m hardware_troubleshooter_arm.build arm.yaml --out DIR [--calibration cal.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_arm_calibration, load_arm_config
from .profile import build_sdk_profile, dump_sdk_profile
from .urdf import build_urdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arm_config")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--calibration", help="measured overrides for this arm")
    args = parser.parse_args(argv)

    config, problems = load_arm_config(args.arm_config)
    calibration, calibration_problems = load_arm_calibration(args.calibration)
    problems += calibration_problems
    if not problems:
        out = Path(args.out)
        urdf, urdf_problems = build_urdf(config, calibration, out)
        profile, profile_problems = build_sdk_profile(config)
        problems = urdf_problems + profile_problems
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    urdf_path = out / f"{config.name}.urdf"
    profile_path = out / f"{config.name}_sdk_profile.yaml"
    urdf_path.write_text(urdf, encoding="utf-8")
    profile_path.write_text(dump_sdk_profile(config, profile), encoding="utf-8")
    print(urdf_path)
    print(profile_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

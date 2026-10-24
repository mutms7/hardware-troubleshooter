"""Test-only values for an imaginary toolhead. They say nothing about the real one."""

import struct
from pathlib import Path

ZERO = [0.0, 0.0, 0.0]

SYNTHETIC_ARM = {
    "name": "left",
    "remove_link_007_meshes": ["meshes/part_026_solid_026.stl", "meshes/part_027_solid_027.stl"],
    "flange": {"xyz_m": [0.0, 0.0, -0.12], "rpy_rad": ZERO},
    "toolhead": {
        "mesh": "toolhead.stl",
        "mesh_scale": [0.001, 0.001, 0.001],
        "joint_origin": {"xyz_m": [0.0, 0.0, -0.01], "rpy_rad": ZERO},
        "axis": [0.0, 1.0, 0.0],
        "lower_rad": -3.0,
        "upper_rad": 3.0,
        "select_tolerance_rad": 0.02,
        "tools": [
            {
                "name": "probe",
                "select_rad": 0.0,
                "tip": {"xyz_m": [0.0, 0.0, -0.05], "rpy_rad": ZERO},
            },
            {
                "name": "clip",
                "select_rad": 1.5,
                "tip": {"xyz_m": [0.0, 0.0, -0.04], "rpy_rad": ZERO},
            },
        ],
    },
    "wrist_camera": {"xyz_m": [0.05, 0.0, -0.08], "rpy_rad": [0.0, 0.3, 0.0]},
    "joint_map": [{"scale": 1.0, "offset_rad": 0.0}] * 7,
    "sdk_joint7": {"direction": 1, "offset": 0.0, "lo": -2.5, "hi": 2.5, "kp": 120, "kd": 3.0},
}

SYNTHETIC_CALIBRATION = {
    "calibration_id": "left-cal-1",
    "arm": "left",
    "wrist_camera": {"xyz_m": [0.051, 0.001, -0.079], "rpy_rad": [0.0, 0.31, 0.0]},
    "tools": {
        "probe": {
            "tip": {"xyz_m": [0.0005, 0.0, -0.0507], "rpy_rad": ZERO},
            "tip_pixel": {"uv": [322.5, 401.0], "standoff_m": 0.04},
        }
    },
}


def write_stl(path: Path) -> None:
    """One-triangle binary STL."""
    triangle = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0)
    path.write_bytes(b"\0" * 80 + struct.pack("<I", 1) + triangle)

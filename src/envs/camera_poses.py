"""Calibrated poses of the three fixed cameras, shared by every task."""

import numpy as np
import sapien
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat

# Rz(-90°), from the calibration frame to the world frame
_R_A2W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)


def _realsense_pose(position: list[float], rpy_degrees: list[float]) -> sapien.Pose:
    roll, pitch, yaw = np.deg2rad(rpy_degrees)
    R_RS2W = _R_A2W @ euler2mat(roll, pitch, yaw, axes="sxyz")
    # RealSense to SAPIEN: SAP+x=RS+z, SAP+y=-RS+x, SAP+z=-RS+y
    R_SAP2W = np.column_stack([R_RS2W[:, 2], -R_RS2W[:, 0], -R_RS2W[:, 1]])
    return sapien.Pose(p=position, q=mat2quat(R_SAP2W))


def fixed_camera_poses() -> tuple[sapien.Pose, sapien.Pose, sapien.Pose]:
    """Return the front (base), left and right camera poses."""
    return (
        _realsense_pose([0.618, -0.005, 0.729], [-127.0, -0.1, 179.9]),
        _realsense_pose([-0.706, 0.670, 0.732], [-131.3, 0.8, -41.2]),
        _realsense_pose([-0.769, -0.803, 0.686], [-134.4, -0.7, 49.5]),
    )

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SceneConfig, SimConfig
import sys
sys.path.insert(0, str(Path(__file__).parent))
from camera_poses import fixed_camera_poses
from custom_table_scene_builder import ROBOT_BASE_POSE, CustomTableSceneBuilder
from distractor_mixin import DistractorMixin
from panda_wristcam_hd import PandaWristCamHD
from put_basket import (
    load_object_half_size,
    load_object_quaternion,
    load_object_footprint_radius,
)


ASSETS_ROOT = Path(__file__).parent.parent.parent / "assets" / "scenes" / "stack_cups"
CUP_NAMES = ["red_cup", "light_blue_cup", "dark_blue_cup"]
OBJECT_SCALES = {
    "red_cup": 0.83,
    "light_blue_cup": 0.8,
    "dark_blue_cup": 0.83,
}


# Center of the cup spawn region, in front of robot and reachable.
SPAWN_CENTER_X = -0.05
SPAWN_CENTER_Y = 0.0
# Pairwise spacing between cups along the y-axis. 5-10 cm range.
MIN_SPACING = 0.17
MAX_SPACING = 0.20
# Small per-cup jitter in x.
CUP_X_JITTER = 0.02


@register_env("StackCups-v1", max_episode_steps=200, override=True)
class StackCupsEnv(DistractorMixin, BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "fetch", "panda_wristcam_hd"]
    agent: Union[Panda, Fetch, PandaWristCamHD]

    _NEUTRAL_QPOS = np.array(
        [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04]
    )

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_scale = dict(OBJECT_SCALES)
        self.assets_root = ASSETS_ROOT
        self.object_names = list(CUP_NAMES)
        self.pick_objects = list(CUP_NAMES)
        self.target_name = CUP_NAMES[0]
        self.object_actors: dict[str, Any] = {}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            # PCM reports false overlaps of about 5 cm on thin convex shapes, which launch objects.
            scene_config=SceneConfig(enable_pcm=False),
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**18,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose_front, pose_left, pose_right = fixed_camera_poses()
        pose_overhead = sapien_utils.look_at(eye=[0, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose_front, width=512, height=288, fov=np.deg2rad(43), near=0.01, far=100),
            CameraConfig("left_camera", pose=pose_left, width=512, height=288, fov=np.deg2rad(43), near=0.01, far=100),
            CameraConfig("right_camera", pose=pose_right, width=512, height=288, fov=np.deg2rad(43), near=0.01, far=100),
            CameraConfig("overhead_camera", pose=pose_overhead, width=512, height=512, fov=np.pi / 2, near=0.01, far=100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=512,
            height=512,
            fov=1,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, ROBOT_BASE_POSE)

    def _load_scene(self, options: dict):
        self.table_scene = CustomTableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        for index, name in enumerate(self.object_names):
            self.object_actors[name] = self._build_object(name, index)
        super()._load_scene(options)

    def _build_object(self, name: str, index: int):
        asset_dir = self.assets_root / name
        visual_glb = str(asset_dir / "visual.glb")
        collision_obj = str(asset_dir / "collision_vhacd.obj")
        object_json = asset_dir / "orientation_result.json"

        builder = self.scene.create_actor_builder()
        scale = self.object_scale[name]
        builder.add_visual_from_file(visual_glb, scale=[scale] * 3)
        builder.add_multiple_convex_collisions_from_file(
            collision_obj, scale=[scale] * 3
        )
        obj = builder.build(name=name)

        mass = 1.0 if name == "dark_blue_cup" else 0.1
        for entity in obj._objs:
            for component in entity.components:
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    component.mass = mass

        obj.set_pose(
            sapien.Pose(
                p=[index * 0.5, 0, load_object_half_size(object_json) * scale]
            )
        )
        return obj

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self._randomize_object_positions()
            self.agent.reset(self._NEUTRAL_QPOS)
        super()._initialize_episode(env_idx, options)

    def _randomize_object_positions(self):
        # Random order so cup colors aren't always in the same y-order.
        order = list(np.random.permutation(self.object_names))
        spacings = np.random.uniform(MIN_SPACING, MAX_SPACING, size=len(order) - 1)
        total = float(np.sum(spacings))
        y_positions = [-total / 2.0]
        for s in spacings:
            y_positions.append(y_positions[-1] + float(s))

        spawn_xy: dict[str, tuple[float, float]] = {}
        for name, y_rel in zip(order, y_positions):
            x = SPAWN_CENTER_X + float(np.random.uniform(-CUP_X_JITTER, CUP_X_JITTER))
            y = SPAWN_CENTER_Y + y_rel
            spawn_xy[name] = (x, y)

        for name, obj in self.object_actors.items():
            object_json = self.assets_root / name / "orientation_result.json"
            half_size = load_object_half_size(object_json) * self.object_scale[name]
            quat = load_object_quaternion(object_json).tolist()
            x, y = spawn_xy[name]

            obj.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor(
                        [[x, y, half_size + 0.001]], dtype=torch.float32, device=self.device
                    ),
                    q=quat,
                )
            )
            for entity in obj._objs:
                for component in entity.components:
                    if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                        component.set_linear_velocity([0, 0, 0])
                        component.set_angular_velocity([0, 0, 0])

        for _ in range(10):
            self.scene.step()

    def evaluate(self):
        # Stacking success (bottom→top): dark_blue → red → light_blue,
        # measured by xy-alignment and ascending z-order.
        positions = {
            name: obj.pose.p[0].cpu().numpy() for name, obj in self.object_actors.items()
        }
        radii = {
            name: load_object_footprint_radius(
                self.assets_root / name / "orientation_result.json"
            ) * self.object_scale[name]
            for name in self.object_names
        }

        bottom, middle, top = "dark_blue_cup", "red_cup", "light_blue_cup"
        xy_mid = float(np.linalg.norm(positions[middle][:2] - positions[bottom][:2]))
        xy_top = float(np.linalg.norm(positions[top][:2] - positions[middle][:2]))
        z_order = (
            positions[bottom][2] < positions[middle][2] < positions[top][2]
        )
        aligned = xy_mid <= radii[bottom] and xy_top <= radii[middle]
        success = bool(z_order and aligned)

        return {
            "success": torch.tensor([success], dtype=torch.bool, device=self.device),
        }

    def _get_obs_extra(self, info: dict):
        return {"tcp_pose": self.agent.tcp.pose.raw_pose}

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

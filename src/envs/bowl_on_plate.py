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


ASSETS_ROOT = Path(__file__).parent.parent.parent / "assets" / "scenes_v2" / "bowl_on_plate"
BOWL_NAME = "bowl"
PLATE_NAME = "plate"
OBJECT_SCALE = 1.0  # the v2 meshes are modelled at real-world size
BOWL_STATIC_FRICTION = 1.0
BOWL_DYNAMIC_FRICTION = 1.0


# v1's distances from the robot base: v1's range (-0.14, -0.09) moved with the base, -0.615 to -0.95.
PLATE_SPAWN_X_RANGE = (-0.475, -0.425)
PLATE_SPAWN_Y_RANGE = (-0.05, 0.05)
BOWL_SIDE_OFFSET = 0.30   # bowl placed at −y of plate by this offset
BOWL_X_JITTER = 0.02
BOWL_Y_JITTER = 0.03


@register_env("BowlOnPlate-v1", max_episode_steps=200, override=True)
class BowlPlateEnv(DistractorMixin, BaseEnv):
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
        object_scale=OBJECT_SCALE,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_scale = float(object_scale)
        self.assets_root = ASSETS_ROOT
        self.object_names = [BOWL_NAME, PLATE_NAME]
        self.pick_objects = [BOWL_NAME]
        self.target_name = PLATE_NAME
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
        self.bowl = self.object_actors[BOWL_NAME]
        self.plate = self.object_actors[PLATE_NAME]
        super()._load_scene(options)

    def _build_object(self, name: str, index: int):
        asset_dir = self.assets_root / name
        visual_glb = str(asset_dir / "visual.glb")
        collision_obj = str(asset_dir / "collision_vhacd.obj")
        object_json = asset_dir / "orientation_result.json"

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(visual_glb, scale=[self.object_scale] * 3)
        material = None  # the scene default
        if name == BOWL_NAME:
            material = sapien.physx.PhysxMaterial(
                static_friction=BOWL_STATIC_FRICTION,
                dynamic_friction=BOWL_DYNAMIC_FRICTION,
                restitution=sapien.physx.get_default_material().restitution,
            )
        builder.add_multiple_convex_collisions_from_file(
            collision_obj, scale=[self.object_scale] * 3, material=material
        )
        obj = builder.build(name=name)

        for entity in obj._objs:
            for component in entity.components:
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    component.mass = 0.2

        obj.set_pose(
            sapien.Pose(
                p=[index * 0.5, 0, load_object_half_size(object_json) * self.object_scale]
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
        plate_x = float(np.random.uniform(*PLATE_SPAWN_X_RANGE))
        plate_y = float(np.random.uniform(*PLATE_SPAWN_Y_RANGE))

        bowl_x = plate_x + float(np.random.uniform(-BOWL_X_JITTER, BOWL_X_JITTER))
        bowl_y = plate_y - BOWL_SIDE_OFFSET + float(
            np.random.uniform(-BOWL_Y_JITTER, BOWL_Y_JITTER)
        )

        spawn_xy = {BOWL_NAME: (bowl_x, bowl_y), PLATE_NAME: (plate_x, plate_y)}

        for name, obj in self.object_actors.items():
            object_json = self.assets_root / name / "orientation_result.json"
            half_size = load_object_half_size(object_json) * self.object_scale
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
        bowl_pos = self.bowl.pose.p[0].cpu().numpy()
        plate_pos = self.plate.pose.p[0].cpu().numpy()

        bowl_half = load_object_half_size(
            self.assets_root / BOWL_NAME / "orientation_result.json"
        ) * self.object_scale
        plate_half = load_object_half_size(
            self.assets_root / PLATE_NAME / "orientation_result.json"
        ) * self.object_scale
        plate_radius = load_object_footprint_radius(
            self.assets_root / PLATE_NAME / "orientation_result.json"
        ) * self.object_scale

        xy_dist = float(np.linalg.norm(bowl_pos[:2] - plate_pos[:2]))
        plate_top_z = float(plate_pos[2] + plate_half)
        bowl_above = float(bowl_pos[2]) >= plate_top_z - 0.01
        bowl_centered = xy_dist <= plate_radius

        success = bowl_above and bowl_centered

        return {
            "success": torch.tensor([success], dtype=torch.bool, device=self.device),
            "bowl_above_plate": torch.tensor([bowl_above], dtype=torch.bool, device=self.device),
            "bowl_centered_on_plate": torch.tensor([bowl_centered], dtype=torch.bool, device=self.device),
        }

    def _get_obs_extra(self, info: dict):
        return {"tcp_pose": self.agent.tcp.pose.raw_pose}

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

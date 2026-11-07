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


ASSETS_ROOT = Path(__file__).parent.parent.parent / "assets" / "scenes" / "open_lid"
LID_NAME = "lid"
BASKET_NAME = "basket"
LID_SCALE = 1.44
BASKET_SCALE = 2.0


BASKET_SPAWN_X_RANGE = (-0.10, 0.05)
BASKET_SPAWN_Y_RANGE = (-0.20, 0.20)


@register_env("OpenLid-v1", max_episode_steps=200, override=True)
class OpenLidEnv(DistractorMixin, BaseEnv):
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
        lid_scale=LID_SCALE,
        basket_scale=BASKET_SCALE,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_scale = {LID_NAME: float(lid_scale), BASKET_NAME: float(basket_scale)}
        self.assets_root = ASSETS_ROOT
        self.object_names = [LID_NAME, BASKET_NAME]
        self.pick_objects = [LID_NAME]
        self.target_name = BASKET_NAME
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
        self.object_actors[BASKET_NAME] = self._build_basket()
        self.object_actors[LID_NAME] = self._build_lid()
        super()._load_scene(options)

    def _build_basket(self):
        asset_dir = self.assets_root / BASKET_NAME
        builder = self.scene.create_actor_builder()
        scale = self.object_scale[BASKET_NAME]
        basket_json = self.assets_root / BASKET_NAME / "orientation_result.json"
        basket_half = load_object_half_size(basket_json) * scale
        basket_half_phys = basket_half - 0.005 * scale  # strip placement margin → matches visual mesh
        basket_radius = load_object_footprint_radius(basket_json) * scale
        builder.add_visual_from_file(str(asset_dir / "visual.glb"), scale=[scale] * 3)
        # Solid box collision instead of VHACD: the strainer VHACD has gaps between
        # convex pieces that the lid slips through. A single solid box gives the lid
        # one clean flat surface to rest on.
        builder.add_box_collision(half_size=[basket_radius, basket_half_phys, basket_radius])
        obj = builder.build_kinematic(name=BASKET_NAME)
        return obj

    def _build_lid(self):
        asset_dir = self.assets_root / LID_NAME
        builder = self.scene.create_actor_builder()
        scale = self.object_scale[LID_NAME]
        builder.add_visual_from_file(str(asset_dir / "visual.glb"), scale=[scale] * 3)
        builder.add_multiple_convex_collisions_from_file(
            str(asset_dir / "collision_vhacd.obj"), scale=[scale] * 3
        )
        obj = builder.build(name=LID_NAME)
        for entity in obj._objs:
            for component in entity.components:
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    component.mass = 0.2
        return obj

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self._place_objects()
            self.agent.reset(self._NEUTRAL_QPOS)
        super()._initialize_episode(env_idx, options)

    def _place_objects(self):
        basket_json = self.assets_root / BASKET_NAME / "orientation_result.json"
        lid_json = self.assets_root / LID_NAME / "orientation_result.json"

        basket_half = load_object_half_size(basket_json) * self.object_scale[BASKET_NAME]
        lid_half = load_object_half_size(lid_json) * self.object_scale[LID_NAME]

        basket_quat = load_object_quaternion(basket_json).tolist()
        lid_quat = load_object_quaternion(lid_json).tolist()

        basket_x = float(np.random.uniform(*BASKET_SPAWN_X_RANGE))
        basket_y = float(np.random.uniform(*BASKET_SPAWN_Y_RANGE))
        basket_z = basket_half + 0.001

        # Lid starts centered on the basket, 1 cm above the basket top.
        lid_z = basket_z + basket_half + lid_half + 0.10

        basket_obj = self.object_actors[BASKET_NAME]
        lid_obj = self.object_actors[LID_NAME]

        basket_obj.set_pose(
            Pose.create_from_pq(
                p=torch.tensor(
                    [[basket_x, basket_y, basket_z]], dtype=torch.float32, device=self.device
                ),
                q=basket_quat,
            )
        )
        lid_obj.set_pose(
            Pose.create_from_pq(
                p=torch.tensor(
                    [[basket_x, basket_y, lid_z]], dtype=torch.float32, device=self.device
                ),
                q=lid_quat,
            )
        )
        for entity in lid_obj._objs:
            for component in entity.components:
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    component.set_linear_velocity([0, 0, 0])
                    component.set_angular_velocity([0, 0, 0])

        for _ in range(60):
            self.scene.step()

    def evaluate(self):
        lid_pos = self.object_actors[LID_NAME].pose.p[0].cpu().numpy()
        basket_pos = self.object_actors[BASKET_NAME].pose.p[0].cpu().numpy()

        # Success: lid has been placed on the table.
        # Lid on table z ≈ 0.008 m < basket center z ≈ 0.024 m;
        # lid resting on basket z ≈ 0.051 m > basket center → False.
        success = bool(float(lid_pos[2]) < float(basket_pos[2]))

        return {
            "success": torch.tensor([success], dtype=torch.bool, device=self.device),
        }

    def _get_obs_extra(self, info: dict):
        return {"tcp_pose": self.agent.tcp.pose.raw_pose}

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SceneConfig, SimConfig
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from camera_poses import fixed_camera_poses
from custom_table_scene_builder import ROBOT_BASE_POSE, CustomTableSceneBuilder
from distractor_mixin import DistractorMixin
from panda_wristcam_hd import PandaWristCamHD

CUBE_DIR = Path(__file__).parent.parent.parent / "assets" / "scenes_v2" / "push_cube" / "cube"
# Keeps v1's distances from the robot: the base moved from x = -0.615 to -0.95.
SPAWN_X_SHIFT = -0.335


@register_env("PushCube-v2", max_episode_steps=200, override=True)
class PushCubeEnvV2(DistractorMixin, BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam_hd"]
    agent: Union[Panda, PandaWristCamHD]

    goal_radius = 0.05
    cube_half_size = 0.02

    # Push distance: cube is pushed this far forward along +X from its spawn position.
    PUSH_DISTANCE = 0.15

    _NEUTRAL_QPOS = np.array(
        [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04]
    )

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            # PCM reports false overlaps of about 5 cm on thin convex shapes, which launch objects.
            scene_config=SceneConfig(enable_pcm=False),
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
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
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(str(CUBE_DIR / "collision_vhacd.obj"))
        builder.add_visual_from_file(str(CUBE_DIR / "visual.glb"))
        builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
        self.obj = builder.build(name="cube")

        super()._load_scene(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 0] += SPAWN_X_SHIFT
            xyz[..., 2] = self.cube_half_size
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))

            target_xyz = xyz.clone()
            target_xyz[..., 0] += self.PUSH_DISTANCE
            target_xyz[..., 2] = 1e-3
            self.goal_pos = target_xyz

            self.agent.reset(self._NEUTRAL_QPOS)
        super()._initialize_episode(env_idx, options)

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_pos[..., :2], axis=1
            )
            < self.goal_radius
        ) & (self.obj.pose.p[..., 2] < self.cube_half_size + 5e-3)
        return {"success": is_obj_placed}

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                goal_pos=self.goal_pos,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tcp_to_obj = self.obj.pose.p - self.agent.tcp_pose.p
        reaching_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_to_obj, axis=1))
        reward = reaching_reward

        obj_to_goal = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_pos[..., :2], axis=1
        )
        push_progress = torch.clamp(1 - obj_to_goal / self.PUSH_DISTANCE, 0.0, 1.0)
        reward += push_progress

        reward[info["success"]] = 4
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4

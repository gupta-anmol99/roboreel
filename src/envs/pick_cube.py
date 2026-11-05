from typing import Any, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from distractor_mixin import DistractorMixin
from panda_wristcam_hd import PandaWristCamHD


@register_env("PickCube-v2", max_episode_steps=200)
class PickCubeEnvV2(DistractorMixin, BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam_hd"]
    agent: Union[Panda, PandaWristCamHD]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    # LIFT_HEIGHT = 0.30  # cube must be lifted at least this many metres above its spawn z
    LIFT_HEIGHT = 0.03  # For eval

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.cube_init_z: float | None = None  # set each episode after cube is placed
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose_front = sapien_utils.look_at(eye=[0.4, 0, 0.6], target=[-0.1, 0, 0.1])
        pose_left = sapien_utils.look_at(eye=[0.4, 0.4, 0.6], target=[-0.1, 0, 0.1])
        pose_right = sapien_utils.look_at(eye=[0.4, -0.4, 0.6], target=[-0.1, 0, 0.1])
        pose_overhead = sapien_utils.look_at(eye=[0, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose_front,
                width=512,
                height=512,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            ),
            CameraConfig(
                "left_camera",
                pose=pose_left,
                width=512,
                height=512,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            ),
            CameraConfig(
                "right_camera",
                pose=pose_right,
                width=512,
                height=512,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            ),
            CameraConfig(
                "overhead_camera",
                pose=pose_overhead,
                width=512,
                height=512,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            ),
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
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        super()._load_scene(options)

    # Neutral arm pose used to reset the robot at the end of _initialize_episode.
    # Fingers are explicitly set to 0.04 (fully open) so that saved_state always
    # has an open gripper — prevents mplib from seeing a self-colliding start state
    # on every grasp attempt.
    _NEUTRAL_QPOS = np.array(
        [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04]
    )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]

            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # Record the spawn height so evaluate() can measure lift distance.
            self.cube_init_z = float(xyz[0, 2].item())

            # Re-apply the neutral qpos after all scene.step() calls so the robot
            # always starts with a known pose and fully open fingers (0.04 m).
            # Without this, physics drift during initialization leaves the fingers
            # in an indeterminate state which causes mplib to see a self-colliding
            # start state on the first grasp attempt.
            self.agent.reset(self._NEUTRAL_QPOS)
        super()._initialize_episode(env_idx, options)

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.cube)
        cube_z = self.cube.pose.p[:, 2]
        init_z = self.cube_init_z if self.cube_init_z is not None else self.cube_half_size
        is_lifted = (cube_z - init_z) >= self.LIFT_HEIGHT
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_lifted & is_robot_static,
            "is_lifted": is_lifted,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        cube_z = self.cube.pose.p[:, 2]
        init_z = self.cube_init_z if self.cube_init_z is not None else self.cube_half_size
        lift_progress = torch.clamp((cube_z - init_z) / self.LIFT_HEIGHT, 0.0, 1.0)
        reward += lift_progress * is_grasped

        qvel = self.agent.robot.get_qvel()[..., :-2]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_lifted"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


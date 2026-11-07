from typing import Any, Optional, Union

import numpy as np
import sapien
import torch
from loguru import logger

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_articulation_builder
from camera_poses import fixed_camera_poses
from custom_table_scene_builder import ROBOT_BASE_POSE, CustomTableSceneBuilder
from distractor_mixin import DistractorMixin
from panda_wristcam_hd import PandaWristCamHD

ASSET_COLLISION_BIT = 29
SUCCESS_DISPLACEMENT = 0.02  # ~80 % of the 0.1136 m joint range (qmin=-0.030, qmax=0.084)

TOASTER_SPAWN_X = 0.0
TOASTER_SPAWN_Y = 0.30

_NEUTRAL_QPOS = np.array(
    [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04]
)


@register_env("Toaster-v1", max_episode_steps=200)
class ToasterEnv(DistractorMixin, BaseEnv):
    SUPPORTED_ROBOTS = ["fetch", "panda", "panda_wristcam_hd"]
    agent: Union[Fetch, Panda, PandaWristCamHD]

    TRAIN_JSON = ASSET_DIR / "configs/info_toaster.json"

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        robot_init_qpos_noise=0.02,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.TRAIN_JSON)
        self.all_model_ids = np.array(list(train_data.keys()))
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            # PCM reports false overlaps of about 5 cm on thin convex shapes, which launch objects.
            scene_config=SceneConfig(enable_pcm=False),
            spacing=5,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            ),
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
        pose = sapien_utils.look_at(eye=[-0.4, 1.0, 1], target=[0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, ROBOT_BASE_POSE)

    def _load_scene(self, options: dict):
        self.table_scene = CustomTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        sapien.set_log_level("off")
        self._load_asset()
        sapien.set_log_level("warn")
        super()._load_scene(options)

    def _load_asset(self):
        model_ids = np.full(self.num_envs, self.all_model_ids[0])

        self._assets: list[Articulation] = []
        target_links: list[Link] = []

        for i, model_id in enumerate(model_ids):
            model_id = int(model_id)
            asset_builder = get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}"
            )
            asset_builder.set_scene_idxs(scene_idxs=[i])
            asset_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
            asset = asset_builder.build(name=f"toaster_{model_id}_env{i}")

            self.remove_from_state_dict_registry(asset)

            for link in asset.links:
                link.set_collision_group_bit(group=2, bit_idx=ASSET_COLLISION_BIT, bit=1)

            # Spring-load the lever so gravity can't press it on its own.
            # The joint axis maps to world -Z, so gravity increases qpos naturally.
            # Stiffness counteracts that; the robot must overcome it to press the lever.
            for active_joint in asset.active_joints:
                active_joint.set_drive_properties(stiffness=50, damping=10)

            self._assets.append(asset)

            # Find the single prismatic (slider) link
            prismatic_links: list[tuple[int, Link]] = []
            dof_count = 0
            for link, joint in zip(asset.links, asset.joints):
                if joint.type[0] == "fixed":
                    continue
                if joint.type[0] == "prismatic":
                    prismatic_links.append((dof_count, link))
                dof_count += 1

            assert len(prismatic_links) > 0, f"Model {model_id} has no prismatic joints"
            self._slider_dof_idx, target_link = prismatic_links[0]
            logger.debug(f"Toaster slider: {target_link.name}, DOF index: {self._slider_dof_idx}")
            target_links.append(target_link)

        self.asset = Articulation.merge(self._assets, name="toaster")
        self.add_to_state_dict_registry(self.asset)
        self.slider_link = Link.merge(target_links, name="target_link")
        self.slider_link_goal = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 1],
            name="link_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )

    def _after_reconfigure(self, options):
        self.asset_zs = []
        for asset in self._assets:
            collision_mesh = asset.get_first_collision_mesh()
            self.asset_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.asset_zs = common.to_tensor(self.asset_zs, device=self.device)

        qlimits = self.slider_link.joint.limits  # [b, 1, 2]
        self.qmin = qlimits[..., 0]
        self.qmax = qlimits[..., 1]

    def slider_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return self.slider_link.pose.p
        return self.slider_link.pose.p[env_idx]

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)

            self.table_scene.initialize(env_idx)
            self.agent.reset(_NEUTRAL_QPOS)

            pos = torch.zeros((b, 3))
            pos[:, 0] = TOASTER_SPAWN_X + (torch.rand(b, device=self.device) * 2 - 1) * 0.07
            pos[:, 1] = TOASTER_SPAWN_Y + (torch.rand(b, device=self.device) * 2 - 1) * 0.40
            pos[:, 2] = self.asset_zs[env_idx] + 0.001
            quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=pos.dtype).repeat(b, 1)
            self.asset.set_pose(Pose.create_from_pq(p=pos, q=quat))

            # Start slider at qmin (lever fully up / unPressed)
            qlimits = self.asset.get_qlimits()
            init_qpos = qlimits[env_idx, :, 0].clone()
            self.asset.set_qpos(init_qpos)
            self.asset.set_qvel(self.asset.qpos[env_idx] * 0)
            for env_i in env_idx.tolist():
                qmin_val = self.qmin[env_i].item()
                for joint in self._assets[env_i].active_joints:
                    joint.set_drive_target(qmin_val)

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            self.slider_link_goal.set_pose(
                Pose.create_from_pq(p=self.slider_link_positions(env_idx))
            )
        super()._initialize_episode(env_idx, options)

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        self.slider_link_goal.set_pose(
            Pose.create_from_pq(p=self.slider_link_positions())
        )
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def evaluate(self):
        current_qpos = self.slider_link.joint.qpos
        displaced_enough = current_qpos >= self.qmin + SUCCESS_DISPLACEMENT
        slider_pos = self.slider_link_positions()

        link_is_static = (
            torch.linalg.norm(self.slider_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.slider_link.linear_velocity, axis=1) <= 0.1)

        return {
            "success": displaced_enough & link_is_static,
            "slider_pos": slider_pos,
            "displaced_enough": displaced_enough,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_slider_pos=info["slider_pos"] - self.agent.tcp.pose.p,
                slider_qpos=self.slider_link.joint.qpos,
                slider_pos=info["slider_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_slider_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["slider_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_slider_dist)

        # Fraction of target displacement achieved: 0 = not moved, 1 = 3cm reached
        displacement = torch.clamp(
            (self.slider_link.joint.qpos - self.qmin) / SUCCESS_DISPLACEMENT,
            min=0.0,
            max=1.0,
        )
        press_reward = 2 * displacement

        # Once the slider moves at all, reaching reward is no longer needed
        reaching_reward[displacement > 0.001] = 2
        press_reward[info["displaced_enough"]] = 3
        reward = reaching_reward + press_reward
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0

from typing import Any, Optional, Union

import numpy as np
import sapien
import torch
from loguru import logger

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_articulation_builder
from distractor_mixin import DistractorMixin
from panda_wristcam_hd import PandaWristCamHD

ASSET_COLLISION_BIT = 29
CLOSE_THRESHOLD = 0.05  # drawer is considered closed when within 1 cm of qmin

# Added spawn constants so the drawer starts to the robot's right and rotated
# 90 degrees clockwise around the z-axis.
# DRAWER_SPAWN_X = 0.1
# DRAWER_SPAWN_Y = -0.40
# DRAWER_ROTATION_QUAT = [np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)]

# Neutral joint configuration for Panda (used to reset robot before sampling TCP position)
_NEUTRAL_QPOS = np.array(
    [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04]
)

# Annular sector spawn region (relative to robot TCP in neutral pose)
_SPAWN_R_MIN = 0.50   # inner radius (m)
_SPAWN_R_MAX = 0.70   # outer radius (m)
_SPAWN_HALF_ARC = np.pi / 4  # ±45° from forward (+x) direction


@register_env("CloseDrawer-v1", max_episode_steps=200, override=True)
class CloseDrawerEnv(DistractorMixin, BaseEnv):
    SUPPORTED_ROBOTS = ["fetch", "panda", "panda_wristcam_hd"]
    agent: Union[Fetch, Panda, PandaWristCamHD]


    # success_threshold = 0.8  # fraction of total drawer travel required for success
    success_threshold = 0.5  # for eval

    def __init__(
            self,
            *args,
            robot_uids="fetch",
            robot_init_qpos_noise=0.02,
            reconfiguration_freq=None,
            num_envs=1,
            **kwargs,
    ):
        logger.info(f"CloseDrawerEnv.__init__ called with robot_uids={robot_uids}")

        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.joint_types = ["prismatic"]
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
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
            spacing=5,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose_front = sapien_utils.look_at(eye=[0.4, 0, 0.6], target=[-0.1, 0, 0.1])
        pose_left = sapien_utils.look_at(eye=[0.4, 0.4, 0.6], target=[-0.1, 0, 0.1])
        pose_right = sapien_utils.look_at(eye=[0.4, -0.4, 0.6], target=[-0.1, 0, 0.1])
        pose_overhead = sapien_utils.look_at(eye=[0, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose_front, width=512, height=512, fov=np.pi / 2, near=0.01, far=100),
            CameraConfig("left_camera", pose=pose_left, width=512, height=512, fov=np.pi / 2, near=0.01, far=100),
            CameraConfig("right_camera", pose=pose_right, width=512, height=512, fov=np.pi / 2, near=0.01, far=100),
            CameraConfig("overhead_camera", pose=pose_overhead, width=512, height=512, fov=np.pi / 2, near=0.01, far=100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        sapien.set_log_level("off")
        self._load_asset(self.joint_types)
        sapien.set_log_level("warn")
        super()._load_scene(options)

    _MODEL_ID = "44962"

    def _load_asset(self, joint_types: list[str]):
        model_ids = np.full(self.num_envs, self._MODEL_ID)

        self._assets: list[Articulation] = []
        self._prismatic_links_per_env: list[list[Link]] = []  # all prismatic links per env
        self._model_ids: list[int] = []
        target_links: list[Link] = []

        for i, model_id in enumerate(model_ids):
            model_id = int(model_id)
            self._model_ids.append(model_id)
            logger.info(f"[CloseDrawer] env{i}: loading cabinet model_id={model_id}")
            asset_builder = get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}"
            )
            asset_builder.set_scene_idxs(scene_idxs=[i])
            asset_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
            asset = asset_builder.build(name=f"drawer_{model_id}_env{i}")

            self.remove_from_state_dict_registry(asset)

            for link in asset.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=ASSET_COLLISION_BIT, bit=1
                )
            self._assets.append(asset)

            # Collect all prismatic links with their DOF indices.
            # Fixed joints do not contribute DOFs; count only active joints.
            prismatic_links_with_dof: list[tuple[int, Link]] = []
            dof_count = 0
            for link, joint in zip(asset.links, asset.joints):
                logger.debug(f"link {link.name} joint {joint.name} type {joint.type}")
                if joint.type[0] == "fixed":
                    continue
                if joint.type[0] in joint_types:
                    prismatic_links_with_dof.append((dof_count, link))
                dof_count += 1

            self._prismatic_dof_indices = [idx for idx, _ in prismatic_links_with_dof]
            self._prismatic_links_per_env.append([link for _, link in prismatic_links_with_dof])
            logger.debug(f"Prismatic DOFs: {self._prismatic_dof_indices}")

            # Default handle link (will be overridden per episode in _initialize_episode)
            target_links.append(prismatic_links_with_dof[-1][1])

        self.asset = Articulation.merge(self._assets, name="drawer")
        self.add_to_state_dict_registry(self.asset)

        # Pre-build one merged Link per drawer slot so we can switch cheaply
        # in _initialize_episode without calling Link.merge dynamically.
        n_slots = len(self._prismatic_dof_indices)
        logger.info(f"[CloseDrawer] cabinet has {n_slots} prismatic drawer(s), DOF indices: {self._prismatic_dof_indices}")
        self._handle_links: list[Link] = []
        for slot in range(n_slots):
            slot_links = [self._prismatic_links_per_env[i][slot] for i in range(len(self._assets))]
            merged = Link.merge(slot_links, name=f"target_link_slot{slot}")
            logger.info(f"[CloseDrawer] slot {slot}: link name={slot_links[0].name}")
            self._handle_links.append(merged)

        self.handle_link = self._handle_links[-1]  # default: top drawer
        self.handle_link_goal = actors.build_sphere(
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

        target_qlimits = self.handle_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        # Target: fully closed (qmin)
        self.target_qpos = qmin
        # Track open range for reward shaping
        self.qmin = qmin
        self.qmax = qmax

    def handle_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return self.handle_link.pose.p
        return self.handle_link.pose.p[env_idx]

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)

            self.table_scene.initialize(env_idx)

            # Reset robot to neutral pose first so we can read the TCP position.
            # The TCP world-frame xy serves as the center of the spawn region.
            self.agent.reset(_NEUTRAL_QPOS)
            tcp_pos = self.agent.tcp.pose.p  # shape [num_envs, 3]
            tcp_xy = tcp_pos[env_idx, :2]    # [b, 2] — only x/y needed

            # Sample spawn positions inside an annular sector (fan-shaped region):
            #   - radial distance d  ~ Uniform[_SPAWN_R_MIN, _SPAWN_R_MAX]
            #   - angular offset  θ  ~ Uniform[-_SPAWN_HALF_ARC, +_SPAWN_HALF_ARC]
            # The sector is centred on the robot's forward (+x) direction so
            # the drawer always appears in front of the arm.
            rng = self._batched_episode_rng
            d = torch.tensor(
                rng.uniform(_SPAWN_R_MIN, _SPAWN_R_MAX, size=(b,)),
                device=self.device, dtype=torch.float32,
            )
            self.drawer_distance = d
            theta = torch.tensor(
                rng.uniform(-_SPAWN_HALF_ARC, _SPAWN_HALF_ARC, size=(b,)),
                device=self.device, dtype=torch.float32,
            )

            # Drawer xy position: TCP_xy + d * [cos θ, sin θ]
            # (θ=0 is straight ahead, θ>0 is left, θ<0 is right)
            pos = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
            pos[:, 0] = tcp_xy[:, 0] + d * torch.cos(theta)
            pos[:, 1] = tcp_xy[:, 1] + d * torch.sin(theta)
            pos[:, 2] = self.asset_zs[env_idx] + 0.001

            # Rotation quaternion (wxyz) around z-axis by angle θ:
            #   θ: aligns the drawer's local +x axis toward the radial direction,
            #      which (for this asset) is the face-forward direction pointing at TCP
            # q = [cos(α/2), 0, 0, sin(α/2)]  where α = θ
            alpha = theta
            half_alpha = alpha / 2.0
            quat = torch.zeros((b, 4), device=self.device, dtype=torch.float32)
            quat[:, 0] = torch.cos(half_alpha)   # w
            quat[:, 3] = torch.sin(half_alpha)   # z
            self.asset.set_pose(Pose.create_from_pq(p=pos, q=quat))

            n_drawers = len(self._prismatic_dof_indices)
            # selected = int(self._batched_episode_rng.randint(0, n_drawers, size=(1,))[0])
            selected = 0  # always bottom drawer
            self._active_slot = selected
            self._active_dof_idx = self._prismatic_dof_indices[selected]
            logger.info(f"[CloseDrawer] opening drawer slot {selected} (DOF index={self._active_dof_idx}) out of {n_drawers}")

            # Switch to the pre-built merged link for the selected drawer slot.
            self.handle_link = self._handle_links[selected]

            # Update per-episode qpos limits so reward/evaluate use the right joint.
            target_qlimits = self.handle_link.joint.limits  # [b, 1, 2]
            self.target_qpos = target_qlimits[..., 0]
            self.qmin = target_qlimits[..., 0]
            self.qmax = target_qlimits[..., 1]

            # Start all drawers closed; open only the selected one.
            qlimits = self.asset.get_qlimits()  # [b, max_dof, 2]
            init_qpos = qlimits[env_idx, :, 0].clone()  # all at qmin (closed)
            init_qpos[:, self._active_dof_idx] = qlimits[env_idx, self._active_dof_idx, 1]
            self.asset.set_qpos(init_qpos)
            self.asset.set_qvel(self.asset.qpos[env_idx] * 0)
            # Record the initial (fully open) qpos for the active drawer joint.
            self.initial_qpos = self.qmax.clone()

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            # Log after kinematics refresh so positions reflect the actual physics state.
            logger.info(f"[CloseDrawer] selected handle link pos: {self.handle_link.pose.p}")
            for slot_i, hl in enumerate(self._handle_links):
                logger.info(f"[CloseDrawer] slot {slot_i} link pos: {hl.pose.p}")

            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )
        super()._initialize_episode(env_idx, options)

    def get_state_dict(self):
        state = super().get_state_dict()
        state["active_slot"] = self._active_slot
        state["active_dof_idx"] = self._active_dof_idx
        return state

    def set_state_dict(self, state):
        super().set_state_dict(state)
        self._active_slot = int(state["active_slot"])
        self._active_dof_idx = int(state["active_dof_idx"])
        self.handle_link = self._handle_links[self._active_slot]
        target_qlimits = self.handle_link.joint.limits  # [b, 1, 2]
        self.target_qpos = target_qlimits[..., 0]
        self.qmin = target_qlimits[..., 0]
        self.qmax = target_qlimits[..., 1]
        self.initial_qpos = self.qmax.clone()

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        self.handle_link_goal.set_pose(
            Pose.create_from_pq(p=self.handle_link_positions())
        )
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def evaluate(self):
        current_qpos = self.asset.get_qpos().clone()[:, self._active_dof_idx]

        target = self.target_qpos
        initial = self.initial_qpos
        threshold = torch.abs(self.success_threshold * (target - initial))
        # logger.info(f"[evaluate] initial_qpos={initial.tolist()} target_qpos={target.tolist()} current_qpos={current_qpos.tolist()} threshold={threshold.tolist()}")
        close_enough = torch.abs(current_qpos - initial) >= threshold
        # logger.info(f"[evaluate] close_enough={close_enough.tolist()}")
        handle_link_pos = self.handle_link_positions()

        link_is_static = (
            torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1)
        return {
            "success": close_enough,
            "movements": torch.abs(current_qpos - initial),
            "threshold": threshold,
            "handle_link_pos": handle_link_pos,
            "close_enough": close_enough,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if "state" in self.obs_mode:
            obs.update(
                tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
                target_link_qpos=self.handle_link.joint.qpos,
                target_handle_pos=info["handle_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["handle_link_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_handle_dist)

        # Fraction of travel remaining to close: 0 = fully closed, 1 = fully open
        open_range = self.qmax - self.qmin  # total travel distance
        amount_to_close_left = torch.clamp(
            (self.handle_link.joint.qpos - self.target_qpos) / open_range, min=0.0, max=1.0
        )
        close_reward = 2 * (1 - amount_to_close_left)

        # Once the drawer moves even a little, reaching reward is no longer needed
        reaching_reward[amount_to_close_left < 0.999] = 2
        close_reward[info["close_enough"]] = 3
        reward = reaching_reward + close_reward
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

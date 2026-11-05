from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import sapien
import torch
from loguru import logger
from scipy.spatial.transform import Rotation as Rscipy

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import get_articulation_builder
from distractor_mixin import DistractorMixin
from panda_wristcam_hd import PandaWristCamHD

ASSET_COLLISION_BIT = 29
ASSETS_ROOT = Path(__file__).parent.parent.parent / "assets" / "scenes" / "object_in_drawer"
OBJECT_NAME = "object1"
OBJECT_SCALE = 0.8

_NEUTRAL_QPOS = np.array(
    [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04]
)

_DRAWER_THETA = np.pi / 2   # drawer to the left (+y), opening perpendicular to robot
_DRAWER_DIST  = 0.35        # distance from TCP to drawer centre
_OBJECT_SIDE_OFFSET = 0.30  # object distance in −y from TCP (right side of robot)
_OBJECT_IN_DRAWER_Z_THRESHOLD = 0.12


def _load_object_metadata(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)
    return next(iter(data.values()))


def _load_object_half_size(json_path: Path) -> float:
    meta = _load_object_metadata(json_path)
    return float(meta["object_half_size"]) + 0.005


def _load_object_quaternion(json_path: Path) -> np.ndarray:
    meta = _load_object_metadata(json_path)
    rot = np.array(meta["rotation_matrix"])
    q_xyzw = Rscipy.from_matrix(rot).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


@register_env("ObjectInDrawer-v1", max_episode_steps=400, override=True)
class ObjectInDrawerEnv(DistractorMixin, BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "fetch", "panda_wristcam_hd"]
    agent: Union[Panda, Fetch, PandaWristCamHD]


    # success_threshold = 0.8  # fraction of total drawer travel required for success
    success_threshold = 0.5  # fraction of total drawer travel required for success
    # success_threshold = 0.75  # for eval

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_scale = OBJECT_SCALE
        self.assets_root = ASSETS_ROOT

        self.joint_types = ["prismatic"]

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
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        sapien.set_log_level("off")
        self._load_asset(self.joint_types)
        sapien.set_log_level("warn")
        self._load_object_actor()
        super()._load_scene(options)

    _MODEL_ID = "44962"

    def _load_asset(self, joint_types: list[str]):
        model_ids = np.full(self.num_envs, self._MODEL_ID)

        self._assets: list[Articulation] = []
        self._prismatic_links_per_env: list[list[Link]] = []
        self._model_ids: list[int] = []

        for i, model_id in enumerate(model_ids):
            model_id = int(model_id)
            self._model_ids.append(model_id)
            asset_builder = get_articulation_builder(self.scene, f"partnet-mobility:{model_id}")
            asset_builder.set_scene_idxs(scene_idxs=[i])
            asset_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
            asset = asset_builder.build(name=f"drawer_{model_id}_env{i}")
            self.remove_from_state_dict_registry(asset)

            for link in asset.links:
                link.set_collision_group_bit(group=2, bit_idx=ASSET_COLLISION_BIT, bit=1)
            self._assets.append(asset)

            prismatic_links_with_dof: list[tuple[int, Link]] = []
            dof_count = 0
            for link, joint in zip(asset.links, asset.joints):
                if joint.type[0] == "fixed":
                    continue
                if joint.type[0] in joint_types:
                    prismatic_links_with_dof.append((dof_count, link))
                dof_count += 1

            self._prismatic_dof_indices = [idx for idx, _ in prismatic_links_with_dof]
            self._prismatic_links_per_env.append([link for _, link in prismatic_links_with_dof])

        self.asset = Articulation.merge(self._assets, name="drawer")
        self.add_to_state_dict_registry(self.asset)

        n_slots = len(self._prismatic_dof_indices)
        self._handle_links: list[Link] = []
        for slot in range(n_slots):
            slot_links = [self._prismatic_links_per_env[i][slot] for i in range(len(self._assets))]
            merged = Link.merge(slot_links, name=f"target_link_slot{slot}")
            self._handle_links.append(merged)

        self.handle_link = self._handle_links[-1]
        self.handle_link_goal = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 1],
            name="link_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )

    def _load_object_actor(self):
        object_json = self.assets_root / OBJECT_NAME / "orientation_result.json"
        asset_dir = self.assets_root / OBJECT_NAME

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(str(asset_dir / "visual.glb"), scale=[self.object_scale] * 3)
        builder.add_multiple_convex_collisions_from_file(
            str(asset_dir / "collision_vhacd.obj"), scale=[self.object_scale] * 3
        )
        self.object = builder.build(name=OBJECT_NAME)

        for entity in self.object._objs:
            for component in entity.components:
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    component.mass = 0.2

        self.object_actors = {OBJECT_NAME: self.object}
        self.object.set_pose(sapien.Pose(p=[10, 10, 0.05]))

    def _after_reconfigure(self, options):
        self.asset_zs = []
        for asset in self._assets:
            collision_mesh = asset.get_first_collision_mesh()
            self.asset_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.asset_zs = common.to_tensor(self.asset_zs, device=self.device)

        target_qlimits = self.handle_link.joint.limits
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin
        self.qmin = qmin
        self.qmax = qmax

    def _refresh_kinematics(self):
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()

    def handle_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return self.handle_link.pose.p
        return self.handle_link.pose.p[env_idx]

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)

            self.table_scene.initialize(env_idx)
            self.agent.reset(_NEUTRAL_QPOS)
            tcp_pos = self.agent.tcp.pose.p
            tcp_xy = tcp_pos[env_idx, :2]

            rng = self._batched_episode_rng

            # Fixed placement: drawer to the left of the robot, opening in +y direction.
            # theta = π/2  →  close-axis = [0, +1],  handle sticks out in −y toward robot.
            theta = torch.full((b,), _DRAWER_THETA, device=self.device, dtype=torch.float32)
            d     = torch.full((b,), _DRAWER_DIST,  device=self.device, dtype=torch.float32)
            self.drawer_distance = d

            drawer_xy_jitter = (torch.rand((b, 2), device=self.device, dtype=torch.float32) - 0.5) * 0.10
            pos = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
            pos[:, 0] = tcp_xy[:, 0] + d * torch.cos(theta) + 0.25 + drawer_xy_jitter[:, 0]
            pos[:, 1] = tcp_xy[:, 1] + d * torch.sin(theta) + 0.10 + drawer_xy_jitter[:, 1]
            pos[:, 2] = self.asset_zs[env_idx] + 0.001

            half_alpha = theta / 2.0
            quat = torch.zeros((b, 4), device=self.device, dtype=torch.float32)
            quat[:, 0] = torch.cos(half_alpha)
            quat[:, 3] = torch.sin(half_alpha)
            self.asset.set_pose(Pose.create_from_pq(p=pos, q=quat))

            n_drawers = len(self._prismatic_dof_indices)
            selected = 0  # always bottom drawer
            self._active_slot = selected
            self._active_dof_idx = self._prismatic_dof_indices[selected]
            self.handle_link = self._handle_links[selected]

            target_qlimits = self.handle_link.joint.limits
            self.target_qpos = target_qlimits[..., 0]
            self.qmin = target_qlimits[..., 0]
            self.qmax = target_qlimits[..., 1]

            qlimits = self.asset.get_qlimits()
            init_qpos = qlimits[env_idx, :, 0].clone()
            init_qpos[:, self._active_dof_idx] = qlimits[env_idx, self._active_dof_idx, 1]
            self.asset.set_qpos(init_qpos)
            self.asset.set_qvel(self.asset.qpos[env_idx] * 0)
            self.initial_qpos = self.qmax.clone()
            self._refresh_kinematics()

            handle_open_pos = self.handle_link.pose.p[env_idx].clone()  # [b, 3]

            # Object on the right side of the robot (−y), opposite the drawer (+y).
            object_xy_jitter = (torch.rand((b, 2), device=self.device, dtype=torch.float32) - 0.5) * 0.10
            obj_xy = tcp_xy.clone()
            obj_xy[:, 0] += 0.20 + object_xy_jitter[:, 0]
            obj_xy[:, 1] -= _OBJECT_SIDE_OFFSET
            obj_xy[:, 1] += object_xy_jitter[:, 1]

            object_json = self.assets_root / OBJECT_NAME / "orientation_result.json"
            obj_half = _load_object_half_size(object_json) * self.object_scale
            obj_quat = _load_object_quaternion(object_json).tolist()

            obj_pos = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
            obj_pos[:, 0] = obj_xy[:, 0]
            obj_pos[:, 1] = obj_xy[:, 1]
            obj_pos[:, 2] = obj_half + 0.001

            self.object.set_pose(Pose.create_from_pq(p=obj_pos, q=obj_quat))
            for entity in self.object._objs:
                for component in entity.components:
                    if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                        component.set_linear_velocity([0, 0, 0])
                        component.set_angular_velocity([0, 0, 0])

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
        target_qlimits = self.handle_link.joint.limits
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
        threshold = torch.abs(self.success_threshold * (self.target_qpos - self.initial_qpos))
        drawer_closed = torch.abs(current_qpos - self.initial_qpos) >= threshold

        obj_pos = self.object.pose.p
        handle_pos = self.handle_link.pose.p
        obj_elevated = obj_pos[:, 2] > _OBJECT_IN_DRAWER_Z_THRESHOLD
        obj_near_drawer = torch.linalg.norm(obj_pos[:, :2] - handle_pos[:, :2], dim=1) < 0.30
        object_in_drawer = obj_elevated & obj_near_drawer

        return {
            "success": drawer_closed & object_in_drawer,
            "drawer_closed": drawer_closed,
            "object_in_drawer": object_in_drawer,
            "handle_link_pos": handle_pos,
            "movements": torch.abs(current_qpos - self.initial_qpos),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
                tcp_to_object_pos=self.object.pose.p - self.agent.tcp.pose.p,
                target_link_qpos=self.handle_link.joint.qpos,
                handle_link_pos=info["handle_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj = torch.linalg.norm(self.agent.tcp.pose.p - self.object.pose.p, dim=1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_obj)

        obj_in_drawer = info["object_in_drawer"]
        reach_reward[obj_in_drawer] = 1.0

        # Use .view(-1) to guard against squeeze collapsing [1] → scalar on num_envs=1
        open_range = (self.qmax - self.qmin).view(-1)
        amount_open = torch.clamp(
            (self.handle_link.joint.qpos.view(-1) - self.qmin.view(-1)) / (open_range + 1e-6),
            min=0.0, max=1.0,
        )
        close_reward = 2 * (1 - amount_open)
        close_reward[~obj_in_drawer] = 0.0

        reward = reach_reward + 2 * obj_in_drawer.float() + close_reward
        reward[info["success"].view(-1)] = 6.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0

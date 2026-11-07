from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as Rscipy

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


ASSETS_ROOT = Path(__file__).parent.parent.parent / "assets" / "scenes_v2" / "put_basket"
# Panda and elephant: each is 11.5 cm tall as it rests on the table. Both basket tasks use these.
TOY_SCALES = {"object1": 0.853, "object2": 0.710}
BASKET_SCALE = 1.0  # the v2 basket is modelled at real-world size
BASKET_NAME = "basket"
BASKET_WALL_THICKNESS = 0.018  # straight sections of the wall's collision shape
BASKET_MASS = 0.25  # from the basket's USD file
BASKET_FRICTION = 0.6  # from the basket's USD file
SPAWN_REL_X_MIN = 0.30
SPAWN_REL_X_MAX = 1.05
SPAWN_REL_Y_MIN = -0.55
SPAWN_REL_Y_MAX = 0.55
# v1's distances from the robot base: v1's range (-0.1, -0.05) moved with the base, -0.615 to -0.95.
BASKET_SPAWN_X_MIN = -0.435
BASKET_SPAWN_X_MAX = -0.385
BASKET_SPAWN_Y = 0.0
# Spawn distance sideways from the basket centre, where v1's saved scenes have the settled toys.
SIDE_OBJECT_Y_OFFSETS = {"object1": 0.246, "object2": 0.277}
SIDE_OBJECT_X_OFFSET = 0.04


def load_object_metadata(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)
    return next(iter(data.values()))


def load_object_quaternion(json_path: Path) -> np.ndarray:
    metadata = load_object_metadata(json_path)
    rotation_matrix = np.array(metadata["rotation_matrix"])
    quat_xyzw = Rscipy.from_matrix(rotation_matrix).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def load_object_half_size(json_path: Path) -> float:
    metadata = load_object_metadata(json_path)
    return float(metadata["object_half_size"]) + 0.005


def load_object_footprint_radius(json_path: Path) -> float:
    metadata = load_object_metadata(json_path)
    extents = np.array(metadata["obb_extents"], dtype=np.float64)
    height_axis = int(metadata["face_idx"]) // 2
    footprint_extents = [extent for i, extent in enumerate(extents) if i != height_axis]
    return 0.5 * float(np.sqrt(footprint_extents[0] ** 2 + footprint_extents[1] ** 2))


def basket_half_width(json_path: Path) -> float:
    """Half the width of a square basket, measured to the outside of its walls, before scaling."""
    extents = load_object_metadata(json_path)["obb_extents"]
    return 0.5 * float(max(extents))


def xy_within_basket(basket_pose, point: np.ndarray, half_width: float) -> bool:
    """Whether a world point lies within half_width of the basket centre along both square axes."""
    basket_from_world = np.linalg.inv(basket_pose.to_transformation_matrix()[0].cpu().numpy())
    local = basket_from_world[:3, :3] @ point + basket_from_world[:3, 3]
    # The mesh is Y-up, so X and Z are the basket's horizontal axes.
    return abs(float(local[0])) <= half_width and abs(float(local[2])) <= half_width


def discover_objects(assets_root: Path) -> list[str]:
    names: list[str] = []
    for asset_dir in sorted(assets_root.iterdir()):
        if not asset_dir.is_dir():
            continue
        if (asset_dir / "visual.glb").exists() and (asset_dir / "collision_vhacd.obj").exists():
            names.append(asset_dir.name)
    return names


@register_env("PutBasket-v1", max_episode_steps=200, override=True)
class PutBasketEnv(DistractorMixin, BaseEnv):
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
        basket_name=BASKET_NAME,
        pick_objects=None,
        basket_scale=BASKET_SCALE,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.basket_name = basket_name
        self.basket_scale = float(basket_scale)
        self.object_scale = {**TOY_SCALES, basket_name: self.basket_scale}
        self.assets_root = ASSETS_ROOT
        self.object_names = discover_objects(self.assets_root)
        self.pick_objects = pick_objects or [
            name for name in self.object_names if name != self.basket_name
        ]
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
        self.all_objects = [
            self._build_object(name, index) for index, name in enumerate(self.object_names)
        ]
        self.object_actors = {obj.name: obj for obj in self.all_objects}
        super()._load_scene(options)

    def _build_object(self, name: str, index: int):
        scale = self.object_scale[name]
        asset_dir = self.assets_root / name
        visual_glb = str(asset_dir / "visual.glb")
        collision_obj = str(asset_dir / "collision_vhacd.obj")
        object_json = asset_dir / "orientation_result.json"

        is_basket = name == self.basket_name
        material = None  # the scene default
        if is_basket:
            material = sapien.physx.PhysxMaterial(
                static_friction=BASKET_FRICTION,
                dynamic_friction=BASKET_FRICTION,
                restitution=sapien.physx.get_default_material().restitution,
            )
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(visual_glb, scale=[scale] * 3)
        builder.add_multiple_convex_collisions_from_file(
            collision_obj, scale=[scale] * 3, material=material
        )
        obj = builder.build(name=name)

        for entity in obj._objs:
            for component in entity.components:
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    component.mass = BASKET_MASS if is_basket else 0.2

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
        active_names = {self.basket_name, *self.pick_objects}
        basket_x = float(np.random.uniform(BASKET_SPAWN_X_MIN, BASKET_SPAWN_X_MAX))
        basket_y = BASKET_SPAWN_Y

        side_signs = [-1.0, 1.0]
        side_index = 0

        for obj in self.all_objects:
            scale = self.object_scale[obj.name]
            object_json = self.assets_root / obj.name / "orientation_result.json"
            half_size = load_object_half_size(object_json) * scale
            quat = load_object_quaternion(object_json).tolist()

            if obj.name not in active_names:
                obj.set_pose(
                    Pose.create_from_pq(
                        p=torch.tensor(
                            [[10.0, 10.0, 0.0]], dtype=torch.float32, device=self.device
                        ),
                        q=quat,
                    )
                )
                continue

            if obj.name == self.basket_name:
                x, y = basket_x, basket_y
            else:
                side = side_signs[side_index % len(side_signs)]
                side_index += 1
                y_offset = SIDE_OBJECT_Y_OFFSETS[obj.name]
                x = basket_x - SIDE_OBJECT_X_OFFSET + float(np.random.uniform(-0.05, 0.05))
                y = basket_y + side * y_offset + float(np.random.uniform(-0.01, 0.01))

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

    @property
    def basket_inner_half_width(self) -> float:
        basket_json = self.assets_root / self.basket_name / "orientation_result.json"
        return basket_half_width(basket_json) * self.basket_scale - BASKET_WALL_THICKNESS

    def object_in_basket(self, pick_name: str) -> bool:
        """Whether the object's centre is between the walls and at a height the basket can hold."""
        basket_obj = self.object_actors[self.basket_name]
        basket_z = float(basket_obj.pose.p[0, 2])
        basket_half = load_object_half_size(
            self.assets_root / self.basket_name / "orientation_result.json"
        ) * self.basket_scale
        pick_half = load_object_half_size(
            self.assets_root / pick_name / "orientation_result.json"
        ) * self.object_scale[pick_name]
        pick_pos = self.object_actors[pick_name].pose.p[0].cpu().numpy()
        return (
            xy_within_basket(basket_obj.pose, pick_pos, self.basket_inner_half_width)
            and basket_z - basket_half <= float(pick_pos[2]) <= basket_z + basket_half + pick_half
        )

    def evaluate(self):
        if not self.pick_objects:
            return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

        all_in_basket = all(self.object_in_basket(name) for name in self.pick_objects)
        return {
            "success": torch.tensor([all_in_basket], dtype=torch.bool, device=self.device)
        }

    def _get_obs_extra(self, info: dict):
        return {"tcp_pose": self.agent.tcp.pose.raw_pose}

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

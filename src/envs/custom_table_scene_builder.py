import os.path as osp
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import inspect as _inspect

from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from distractor_mixin import _apply_texture_to_actor


# Upstream table.glb at scale=1.75 spans these world extents (from upstream AABB):
#   x: 0.4688596 - (-0.7402168) = 1.2090764 m
#   y: 1.2030163 - (-1.2148621) = 2.4178784 m
_MESH_X_AT_1_75 = 1.2090764
_MESH_Y_AT_1_75 = 2.4178784

# Robot base: beside the table's near edge (x = -0.8725), 20 cm below its surface.
ROBOT_BASE_POSE = sapien.Pose(p=[-0.95, 0, -0.20])
_FIXED_BASE_ROBOTS = ("panda", "panda_wristcam_hd")

_TABLE_TEXTURE = (
    Path(__file__).parent.parent.parent / "assets" / "distractors" / "textures"
    / "table" / "Final-Texture.png"
)


class CustomTableSceneBuilder(TableSceneBuilder):
    """Table matching the lab table: 150.5 cm (world-x) × 152.4 cm (world-y)."""

    TABLE_X = 1.505   # world-x, 150.5 cm
    TABLE_Y = 1.524   # world-y, 152.4 cm
    TABLE_HEIGHT = 0.9196429

    def build(self):
        builder = self.scene.create_actor_builder()
        table_model_file = str(
            Path(osp.dirname(_inspect.getfile(TableSceneBuilder))) / "assets" / "table.glb"
        )

        # Non-uniform scale so the visual mesh matches the collision box.
        # The mesh local axes align with world axes after two π/2 rotations (= π total),
        # so scale_x drives world-x and scale_y drives world-y.
        scale = [
            1.75 * (self.TABLE_X / _MESH_X_AT_1_75),   # ≈ 2.178
            1.75 * (self.TABLE_Y / _MESH_Y_AT_1_75),   # ≈ 1.103
            1.75,
        ]

        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))

        # Actor is rotated π/2 around z: local-x → world-y, local-y → world-x.
        # So half_size[0] sets world-y and half_size[1] sets world-x.
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, self.TABLE_HEIGHT / 2]),
            half_size=(self.TABLE_Y / 2, self.TABLE_X / 2, self.TABLE_HEIGHT / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=scale, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(
            p=[-0.12, 0, -self.TABLE_HEIGHT], q=euler2quat(0, 0, np.pi / 2)
        )
        self.table = builder.build_kinematic(name="table-workspace")

        self.table_length = self.TABLE_X
        self.table_width = self.TABLE_Y
        self.table_height = self.TABLE_HEIGHT

        floor_width = 500 if self.scene.parallel_in_single_scene else 100
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        self.scene_objects = [self.table, self.ground]

        _apply_texture_to_actor(self.table, str(_TABLE_TEXTURE))

    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        # The parent moves "panda" back to its default base at x = -0.615 on every reset.
        if self.env.robot_uids in _FIXED_BASE_ROBOTS:
            self.env.agent.robot.set_pose(ROBOT_BASE_POSE)

"""DistractorMixin — drop-in domain randomisation for TableSceneBuilder envs.

Usage in any env:
  1. Add DistractorMixin BEFORE BaseEnv in the class declaration.
  2. Call super()._load_scene(options) at the END of _load_scene.
  3. Call super()._initialize_episode(env_idx, options) at the END of
     _initialize_episode (outside any with torch.device block).

Passing distract=True to gym.make() then:
  - Loads a random kinematic distractor object from assets/distractors/objects/
  - Places it each episode >= 10 cm away from every env object (rejection sampling)
  - Randomises the table and floor textures every reconfigure
"""

import json
import random as _random
from pathlib import Path

import numpy as np
import sapien
import sapien.render as sr
from loguru import logger
from scipy.spatial.transform import Rotation as SciRotation

_ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"
_DISTRACTOR_OBJECTS_DIR = _ASSETS_DIR / "distractors" / "objects"
_TABLE_TEXTURE_DIR = _ASSETS_DIR / "distractors" / "textures" / "table"
_FLOOR_TEXTURE_DIR = _ASSETS_DIR / "distractors" / "textures" / "floor"

_TABLE_X_RANGE = (-0.4, 0.4)
_TABLE_Y_RANGE = (-0.4, 0.4)
_CLEARANCE_M = 0.25
_MAX_PLACEMENT_TRIES = 100

_SKIP_NAMES = frozenset({"table-workspace"})
_SKIP_PREFIX = "distractor"
_SKIP_SUBSTRINGS = ("ground", "goal", "panda", "fetch")


def _apply_texture_to_actor(actor, texture_path: str) -> None:
    tex = sr.RenderTexture2D(filename=texture_path, mipmap_levels=4)
    for obj in actor._objs:
        comp = obj.find_component_by_type(sr.RenderBodyComponent)
        if comp is None:
            continue
        for shape in comp.render_shapes:
            if not isinstance(shape, sr.RenderShapeTriangleMesh):
                continue
            for part in shape.parts:
                if part.material is None:
                    continue
                part.material.set_base_color_texture(tex)
                part.material.set_base_color([1.0, 1.0, 1.0, 1.0])


class DistractorMixin:
    def __init__(self, *args, distract: bool = False, **kwargs):
        self.distract = distract
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ scene

    def _load_scene(self, options: dict):
        if self.distract:
            self._distractor_load_object(options)
            # --- TEXTURE RANDOMISATION (disabled) ---
            # To re-enable: uncomment the two lines below.
            # if hasattr(self, "table_scene"):
            #     self._distractor_apply_textures()
            # --- END TEXTURE RANDOMISATION ---

    def _distractor_load_object(self, options: dict | None = None):
        # Eval-replay path: if the caller passes options={"distractor_names":
        # [...]} via env.reset(..., options=...) together with reconfigure=True,
        # honour it exactly instead of sampling. This lets pre-recorded eval
        # banks (eval_env_configs/with_distraction/*.pkl) drive both the
        # distractor *count* and the *asset identities* deterministically, so
        # set_state_dict's name-keyed lookup matches the recorded
        # state["actors"]["distractor_<i>"] slots even when the env's RNG
        # would have sampled a different count/set for the same seed.
        override_names = (options or {}).get("distractor_names")
        obj_dirs = [d for d in _DISTRACTOR_OBJECTS_DIR.iterdir() if d.is_dir()]
        if override_names:
            by_name = {d.name: d for d in obj_dirs}
            try:
                chosen = [by_name[str(n)] for n in override_names]
            except KeyError as e:
                raise KeyError(
                    f"distractor_names override contains unknown asset {e.args[0]!r}; "
                    f"known assets: {sorted(by_name)}"
                ) from None
        else:
            count = _random.randint(1, 2)
            chosen = _random.sample(obj_dirs, min(count, len(obj_dirs)))

        self._distractor_actors = []
        self._distractor_quats = []
        self._distractor_half_sizes = []
        self._distractor_names = []

        for i, obj_dir in enumerate(chosen):
            with open(obj_dir / "orientation_result.json") as f:
                orient = list(json.load(f).values())[0]

            rot_mat = np.array(orient["rotation_matrix"])
            quat_xyzw = SciRotation.from_matrix(rot_mat).as_quat()
            quat = [
                float(quat_xyzw[3]),
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
            ]
            s = float(orient.get("scale", 1.0))
            half_size = float(orient["object_half_size"]) * s
            scale = [s, s, s]
            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(str(obj_dir / "visual.glb"), scale=scale)
            builder.add_multiple_convex_collisions_from_file(str(obj_dir / "collision_vhacd.obj"), scale=scale)
            actor = builder.build_kinematic(name=f"distractor_{i}")

            self._distractor_actors.append(actor)
            self._distractor_quats.append(quat)
            self._distractor_half_sizes.append(half_size)
            self._distractor_names.append(obj_dir.name)
            logger.debug(f"Distractor {i} loaded: {obj_dir.name}")

    def _distractor_apply_textures(self):
        table_textures = list(_TABLE_TEXTURE_DIR.glob("*.png"))
        floor_textures = list(_FLOOR_TEXTURE_DIR.glob("*.png"))
        if table_textures:
            _apply_texture_to_actor(
                self.table_scene.table, str(_random.choice(table_textures))
            )
        if floor_textures:
            _apply_texture_to_actor(
                self.table_scene.ground, str(_random.choice(floor_textures))
            )

    # -------------------------------------------------------- episode init

    def _initialize_episode(self, env_idx, options):
        if self.distract:
            self._distractor_place_with_clearance(env_idx)

    def _distractor_place_with_clearance(self, env_idx):
        env_positions = self._distractor_get_env_positions()
        placed_xy = []

        for idx, (actor, quat, half_size) in enumerate(
            zip(self._distractor_actors, self._distractor_quats, self._distractor_half_sizes)
        ):
            dist_z = half_size + 0.001
            best_pos = None
            best_min_dist = -1.0

            for _ in range(_MAX_PLACEMENT_TRIES):
                x = _random.uniform(*_TABLE_X_RANGE)
                y = _random.uniform(*_TABLE_Y_RANGE)
                all_obstacles = list(env_positions) + placed_xy
                if all_obstacles:
                    min_dist = min(
                        float(np.hypot(x - p[0], y - p[1])) for p in all_obstacles
                    )
                else:
                    min_dist = float("inf")

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_pos = (x, y)

                if min_dist >= _CLEARANCE_M:
                    break

            x, y = best_pos
            actor.set_pose(sapien.Pose(p=[x, y, dist_z], q=quat))
            placed_xy.append((x, y))
            logger.debug(
                f"Distractor {idx} at ({x:.3f}, {y:.3f}, {dist_z:.3f}), "
                f"clearance={best_min_dist:.3f} m"
            )

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def _distractor_get_env_positions(self) -> list:
        def _skip(name: str) -> bool:
            if name in _SKIP_NAMES:
                return True
            if name.startswith(_SKIP_PREFIX):
                return True
            nl = name.lower()
            return any(s in nl for s in _SKIP_SUBSTRINGS)

        positions = []
        for name, actor in self.scene.actors.items():
            if _skip(name):
                continue
            p = actor.pose.p
            positions.append((p[0] if p.ndim > 1 else p).detach().cpu().numpy())

        for name, art in self.scene.articulations.items():
            if _skip(name):
                continue
            p = art.pose.p
            positions.append((p[0] if p.ndim > 1 else p).detach().cpu().numpy())

        return positions

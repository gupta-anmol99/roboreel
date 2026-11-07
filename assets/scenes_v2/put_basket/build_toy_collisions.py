"""Build the panda's and elephant's collision shapes for both basket tasks, then check them.

Each toy gets one boxy convex shape with a flat bottom, placed inside its visual mesh so the fingers look
like they press into the plush. Shapes are written to collision_vhacd.obj (GraspGen samples its point cloud
from this file too), and orientation_result.json is set to the toy's natural resting pose.

Run from the repo root:  python assets/scenes_v2/put_basket/build_toy_collisions.py
Needs numpy, scipy and trimesh.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import ConvexHull

SCENES = Path(__file__).resolve().parent.parent
TASKS = ("put_basket", "empty_basket")

# Keep in sync with TOY_SCALES in src/envs/put_basket.py.
TOYS = {
    "object1": {  # panda, sitting
        "scale": 0.853,
        "up": [0.008, 0.999, -0.043],  # mesh direction that points up when resting: a stable face of the visual's hull
        "forward": [0.0, 0.0, 1.0],
        "yaw_deg": -178.7,  # keeps the heading the toys already had
        # (height, x_min, x_max, y_min, y_max) in cm, in the upright frame: x is the toy's right, y its front.
        # A piece is the convex hull of its stacked rectangles.
        "pieces": [[
            (0, -2.7, 2.7, -4.0, 0.6),
            (5.5, -2.7, 2.7, -4.0, 0.6),
            (7.5, -2.2, 2.2, -2.8, 0.7),
            (11.0, -1.3, 1.6, -1.0, 1.4),
        ]],
    },
    "object2": {  # elephant, on its rear feet and front left foot
        "scale": 0.710,
        "up": [-0.091, 0.408, -0.908],
        "forward": [0.0, 1.0, 0.0],
        "yaw_deg": -147.7,
        "pieces": [[
            (0, -2.2, 2.2, -4.6, 2.0),
            (8.5, -2.2, 2.2, -4.6, 2.0),
            (10.8, -1.2, 1.2, -2.0, 1.8),
        ]],
    },
}
GRIPPER_OPENING_CM = 8.0
MIN_FINGER_CLEARANCE_CM = 0.5  # on each side, across the shape's narrow direction
VOXEL = 0.0015  # m


def load_visual(path: Path) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh", process=False)


def upright_axes(toy: dict) -> np.ndarray:
    """Rows are the toy's right, front and up directions, in mesh coordinates."""
    up = np.array(toy["up"], dtype=float)
    up /= np.linalg.norm(up)
    front = np.array(toy["forward"], dtype=float)
    front -= (front @ up) * up
    front /= np.linalg.norm(front)
    return np.vstack([np.cross(front, up), front, up])


def piece_points(piece) -> np.ndarray:
    points = [[x, y, z] for z, x0, x1, y0, y1 in piece for x in (x0, x1) for y in (y0, y1)]
    return np.array(points, dtype=float) / 100.0


def write_collision(path: Path, pieces: list[np.ndarray], axes: np.ndarray, floor: float, scale: float) -> None:
    lines, offset = [], 1
    for i, points in enumerate(pieces):
        hull = ConvexHull(points)
        used = sorted(set(hull.simplices.ravel()))
        index = {old: new for new, old in enumerate(used)}
        vertices = (points[used] + [0.0, 0.0, floor]) @ axes / scale  # upright frame to unscaled mesh
        lines.append(f"o piece{i}")
        lines += [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in vertices]
        for simplex, equation in zip(hull.simplices, hull.equations):
            tri = [index[j] for j in simplex]
            a, b, c = vertices[tri]
            if np.cross(b - a, c - a) @ (equation[:3] @ axes) < 0:
                tri = [tri[0], tri[2], tri[1]]
            lines.append("f " + " ".join(str(j + offset) for j in tri))
        offset += len(used)
    path.write_text("\n".join(lines) + "\n")


def update_orientation(path: Path, axes: np.ndarray, yaw_deg: float, floor: float, scale: float) -> None:
    data = json.loads(path.read_text())
    meta = next(iter(data.values()))
    yaw = np.radians(yaw_deg)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) @ axes
    transform = np.eye(4)
    transform[:3, :3] = rotation
    meta["rotation_matrix"] = rotation.tolist()
    meta["transform_4x4"] = transform.tolist()
    # load_object_half_size adds 5 mm before scaling; the env spawns the toy 1 mm above the table with this value.
    meta["object_half_size"] = float(-floor / scale - 0.005)
    path.write_text(json.dumps(data, indent=2))


def inside_grid(vertices: np.ndarray, faces: np.ndarray):
    """Voxels the outside can't reach through the surface. The meshes aren't watertight, so this is a flood fill."""
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    low = vertices.min(0) - 3 * VOXEL
    shape = np.ceil((vertices.max(0) + 3 * VOXEL - low) / VOXEL).astype(int)
    samples, _ = trimesh.sample.sample_surface(mesh, int(max(4e6, mesh.area / VOXEL**2 * 8)), seed=0)
    index = np.clip(np.floor((samples - low) / VOXEL).astype(int), 0, shape - 1)
    surface = np.zeros(shape, dtype=bool)
    surface[index[:, 0], index[:, 1], index[:, 2]] = True
    labels, _ = ndimage.label(~surface)
    return labels != labels[0, 0, 0], low


def in_shape(points: np.ndarray, grid: np.ndarray, low: np.ndarray) -> np.ndarray:
    coordinates = [low[i] + VOXEL * np.arange(grid.shape[i]) for i in range(3)]
    cells = np.stack(np.meshgrid(*coordinates, indexing="ij"), -1)
    equations = ConvexHull(points).equations
    return (cells @ equations[:, :3].T + equations[:, 3] <= 1e-9).all(-1)


def check(name: str, toy: dict, pieces: list[np.ndarray], vertices: np.ndarray, faces: np.ndarray) -> None:
    grid, low = inside_grid(vertices, faces)
    shape = np.zeros(grid.shape, dtype=bool)
    for points in pieces:
        shape |= in_shape(points, grid, low)
    distance = ndimage.distance_transform_edt(~grid) * VOXEL
    outside = shape & ~grid
    all_points = np.vstack(pieces)
    width = np.ptp(all_points, axis=0) * 100
    narrow = min(width[0], width[1])
    print(
        f"{name}: {width[0]:.1f} x {width[1]:.1f} x {width[2]:.1f} cm, "
        f"{1 - outside.sum() / shape.sum():.1%} inside the visual, "
        f"sticks out by up to {distance[shape].max() * 1000:.1f} mm, "
        f"fills {(shape & grid).sum() / grid.sum():.1%} of the toy"
    )
    assert abs(all_points[:, 2].min()) < 1e-9, f"{name}: the shape must have its flat bottom on the table"
    assert abs(np.ptp(vertices[:, 2]) - 0.115) < 0.002, f"{name}: the toy must be 11.5 cm tall"
    assert narrow + 2 * MIN_FINGER_CLEARANCE_CM <= GRIPPER_OPENING_CM, f"{name}: too wide for the gripper"
    assert (grid.sum() * VOXEL**3 * 1e6) > 200, f"{name}: flood fill leaked, the toy has no inside"


def main() -> None:
    for name, toy in TOYS.items():
        visual = load_visual(SCENES / TASKS[0] / name / "visual.glb")
        axes = upright_axes(toy)
        vertices = visual.vertices * toy["scale"] @ axes.T
        floor = float(vertices[:, 2].min())
        vertices = vertices - [0.0, 0.0, floor]
        pieces = [piece_points(piece) for piece in toy["pieces"]]
        for task in TASKS:
            asset = SCENES / task / name
            write_collision(asset / "collision_vhacd.obj", pieces, axes, floor, toy["scale"])
            update_orientation(asset / "orientation_result.json", axes, toy["yaw_deg"], floor, toy["scale"])
            written = trimesh.load(asset / "collision_vhacd.obj", force="mesh", process=True)
            assert written.is_watertight and written.volume > 0, f"{task}/{name}: bad collision mesh"
        check(name, toy, pieces, vertices, visual.faces)


if __name__ == "__main__":
    main()

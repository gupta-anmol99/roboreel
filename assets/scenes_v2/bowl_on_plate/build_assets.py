"""Build the v2 bowl and plate assets as surfaces of revolution.

Run from anywhere; writes <name>/visual.glb, collision_vhacd.obj and orientation_result.json next
to this file. Meshes are in metres, Y-up, with the origin at the bounding-box centre, the same
convention as assets/scenes. collision_vhacd.obj is built directly from the profile, not with
V-HACD; it keeps that name because the task loader expects it.

Sources: lab photos and measured dimensions. The plate's height and both wall thicknesses are
estimates, since they weren't measured.
"""

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d

OUT_DIR = Path(__file__).parent
DS = 1e-4  # profile sampling step, metres
SEGMENTS = 128  # mesh divisions around the axis
Y_UP_TO_Z_UP = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]

# --- Bowl: rim 18 cm, base 8 cm, height 7.2 cm (measured).
BOWL_RIM_R, BOWL_BASE_R, BOWL_H = 0.09, 0.04, 0.072
BOWL_WALL = 0.003
# The gripper squeezes this wall, and thin collision pieces get ejected violently when squeezed.
# So the collision wall grows into the cavity; its outer surface still matches the visual one.
BOWL_COLLISION_WALL = BOWL_WALL
# Outer wall traced from the side photo, as (radius / rim radius, height / total height).
BOWL_OUTER_TRACE = np.array(
    [
        (0.585, 0.070), (0.684, 0.140), (0.753, 0.215), (0.803, 0.290), (0.844, 0.355),
        (0.878, 0.430), (0.909, 0.498), (0.931, 0.552), (0.984, 0.790), (1.000, 1.000),
    ]
)
BOWL_SRGB = (178, 178, 174)  # median of unshadowed patches in the photos
BOWL_ROUGHNESS = 0.62

# --- Plate: diameter 28 cm (measured), height 2.5 cm (estimate).
PLATE_R, PLATE_H = 0.14, 0.025
PLATE_FLOOR_Y = 0.0035  # top of the flat centre
PLATE_FLAT_R = 0.092  # edge of the flat centre, from the photo
PLATE_BASE_R = 0.094
PLATE_LIP_R = 0.134  # inner edge of the flat rim
PLATE_SRGB = (226, 221, 206)
PLATE_ROUGHNESS = 0.55
PLATE_LIP_ROUGHNESS = 0.32  # the lip is visibly glossier in the photo


def resample(points, ds=DS):
    points = np.asarray(points, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]
    points = points[keep]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    t = np.linspace(0.0, s[-1], max(int(round(s[-1] / ds)) + 1, 2))
    return np.column_stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])])


def smooth(points, sigma, mirror_start=False, mirror_end=False):
    """Gaussian-smooth a polyline. A mirrored end stays on the axis with a flat tangent."""
    p = resample(points)
    pad = int(4 * sigma / DS) + 1
    head = p[1 : pad + 1][::-1] * ([-1, 1] if mirror_start else [1, 1])
    tail = p[-pad - 1 : -1][::-1] * ([-1, 1] if mirror_end else [1, 1])
    if not mirror_start:
        head = 2 * p[0] - p[1 : pad + 1][::-1]
    if not mirror_end:
        tail = 2 * p[-1] - p[-pad - 1 : -1][::-1]
    padded = np.vstack([head, p, tail])
    out = gaussian_filter1d(padded, sigma / DS, axis=0, mode="nearest")[len(head) : len(head) + len(p)]
    if mirror_start:
        out[0, 0] = 0.0
    if mirror_end:
        out[-1, 0] = 0.0
    return out


def bowl_profile(wall=BOWL_WALL):
    """Closed profile from the bottom axis, up the outside, over the rim, to the inner axis."""

    def build(base_r):
        trace = BOWL_OUTER_TRACE * [BOWL_RIM_R, BOWL_H]
        spline_r = np.r_[base_r, trace[:, 0]]
        spline_y = np.r_[0.0, trace[:, 1]]
        y = np.linspace(0.0, BOWL_H, 800)
        outer_wall = np.column_stack([PchipInterpolator(spline_y, spline_r)(y), y])
        outer = smooth(np.vstack([[0.0, 0.0], outer_wall]), 0.004, mirror_start=True)
        outer[:, 1] = np.maximum(outer[:, 1], 0.0)
        tangent = np.gradient(outer, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        inner = outer + wall * np.column_stack([-tangent[:, 1], tangent[:, 0]])
        inner = inner[(inner[:, 0] >= 0.0) & (inner[:, 1] <= BOWL_H)]
        rim = np.array([[outer[-1, 0], BOWL_H], [inner[-1, 0], BOWL_H]])
        profile = np.vstack([outer, rim, inner[::-1], [[0.0, inner[0, 1]]]])
        profile = smooth(profile, 0.0005, mirror_start=True, mirror_end=True)
        profile[:, 1] -= profile[:, 1].min()
        profile *= [BOWL_RIM_R / profile[:, 0].max(), BOWL_H / profile[:, 1].max()]
        return profile

    base_r = BOWL_BASE_R
    for _ in range(8):  # make the flat contact patch, not the sketch, match the measured base
        profile = build(base_r)
        first_lift = np.argmax(profile[:, 1] > 1e-4)
        base_r += BOWL_BASE_R - profile[first_lift, 0]
    return profile


def plate_profile():
    under = smooth(
        [
            (0.0, 0.0), (PLATE_BASE_R, 0.0), (PLATE_LIP_R - 0.001, PLATE_H - 0.0060),
            (PLATE_R - 0.0015, PLATE_H - 0.0040),
        ],
        0.005,
        mirror_start=True,
    )
    under[:, 1] = np.maximum(under[:, 1], 0.0)
    well = smooth(
        [
            (PLATE_LIP_R, PLATE_H), (PLATE_FLAT_R + 0.004, PLATE_FLOOR_Y + 0.0015),
            (PLATE_FLAT_R - 0.008, PLATE_FLOOR_Y), (0.0, PLATE_FLOOR_Y),
        ],
        0.007,
        mirror_end=True,
    )
    lip = [(PLATE_R, PLATE_H - 0.0035), (PLATE_R, PLATE_H), (PLATE_LIP_R + 0.001, PLATE_H)]
    profile = np.vstack([under, lip, well])
    profile = smooth(profile, 0.0004, mirror_start=True, mirror_end=True)
    profile[:, 1] -= profile[:, 1].min()
    profile *= [PLATE_R / profile[:, 0].max(), PLATE_H / profile[:, 1].max()]
    return profile


def simplify(points, tolerance=2e-5, max_step=0.006):
    """Douglas-Peucker indices into a dense polyline, with a maximum spacing between kept points."""
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        chord = points[b] - points[a]
        length = np.linalg.norm(chord)
        rel = points[a + 1 : b] - points[a]
        dist = np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0]) / max(length, 1e-12)
        i = int(np.argmax(dist))
        if dist[i] > tolerance or length > max_step:
            mid = a + 1 + i if dist[i] > tolerance else (a + b) // 2
            keep[mid] = True
            stack += [(a, mid), (mid, b)]
    return np.flatnonzero(keep)


def revolve(profile):
    """Revolve a dense (r, y) profile. Returns vertices, faces, normals, uv and each ring's v."""
    tangent = np.gradient(profile, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal_2d = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(profile, axis=0), axis=1))]
    arc /= arc[-1]

    rings = simplify(profile)
    theta = np.linspace(0.0, 2 * np.pi, SEGMENTS + 1)
    u, ring = np.meshgrid(theta / (2 * np.pi), np.arange(len(rings)))
    cos, sin = np.cos(u * 2 * np.pi), np.sin(u * 2 * np.pi)
    picked = profile[rings][ring]
    picked_n = normal_2d[rings][ring]
    vertices = np.stack([picked[..., 0] * cos, picked[..., 1], picked[..., 0] * sin], -1).reshape(-1, 3)
    normals = np.stack([picked_n[..., 0] * cos, picked_n[..., 1], picked_n[..., 0] * sin], -1).reshape(-1, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    uv = np.column_stack([u.ravel(), arc[rings][ring].ravel()])

    n = SEGMENTS + 1
    faces = []
    for i in range(len(rings) - 1):
        for j in range(SEGMENTS):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            if profile[rings[i], 0] > 1e-9:
                faces.append((a, c, b))
            if profile[rings[i + 1], 0] > 1e-9:
                faces.append((b, c, d))
    faces = np.array(faces)
    # Wind every face so that it agrees with the analytic outward normal.
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    if np.sum(face_normals * normals[faces].mean(1)) < 0:
        faces = faces[:, ::-1]
    return vertices, faces, normals, uv


def periodic_noise(size, sigma, rng):
    noise = gaussian_filter(rng.standard_normal((size, size)), sigma, mode="wrap")
    return noise / noise.std()


def material(srgb, roughness, rough_band=None, seed=0):
    """Matte moulded plastic: a very faint mottle in colour and sheen, with no dirt or marks."""
    rng = np.random.default_rng(seed)
    size = 1024
    shade = 1.0 * periodic_noise(size, 40, rng) + 0.5 * periodic_noise(size, 3, rng)
    base = np.clip(np.array(srgb, dtype=float)[None, None] + shade[..., None], 0, 255).astype(np.uint8)

    rough = np.full((size, size), float(roughness)) + 0.015 * periodic_noise(size, 12, rng)
    if rough_band is not None:
        # Row 0 is v = 1 in trimesh's image convention.
        v = 1.0 - (np.arange(size) + 0.5) / size
        band = (v >= rough_band[0]) & (v <= rough_band[1])
        rough[band] = rough_band[2] + 0.015 * periodic_noise(size, 12, rng)[band]
    mr = np.zeros((size, size, 3), dtype=np.uint8)
    mr[..., 1] = np.clip(rough * 255, 0, 255).astype(np.uint8)
    return trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(base, "RGB"),
        metallicRoughnessTexture=Image.fromarray(mr, "RGB"),
        metallicFactor=1.0,  # multiplied by the texture's blue channel, which is 0
        roughnessFactor=1.0,
    )


def convex_parts(profile, sectors, levels, center_y_split):
    """Split the shell into convex pieces: one centre disc, then rings of sector-shaped blocks."""
    rim_i = int(np.argmax(profile[:, 1] + 1e-3 * profile[:, 0]))
    outer, inner = profile[: rim_i + 1], profile[rim_i:][::-1]
    start_out = int(np.argmax(outer[:, 1] > center_y_split[0]))
    start_in = int(np.argmax(inner[:, 1] > center_y_split[1]))
    base_r, floor_r = outer[start_out, 0], inner[start_in, 0]
    floor_y = inner[start_in, 1]

    theta = np.linspace(0.0, 2 * np.pi, sectors + 1)

    def ring(r, y):
        return np.column_stack([r * np.cos(theta[:-1]), np.full(sectors, y), r * np.sin(theta[:-1])])

    parts = [np.vstack([ring(base_r, 0.0), ring(floor_r, floor_y)])]
    out_i = np.linspace(start_out, len(outer) - 1, levels + 1).astype(int)
    in_i = np.linspace(start_in, len(inner) - 1, levels + 1).astype(int)
    for k in range(levels):
        section = np.array([outer[out_i[k]], outer[out_i[k + 1]], inner[in_i[k]], inner[in_i[k + 1]]])
        if k == levels - 1:  # the widest point sits just below the rim's top
            section = np.vstack([section, outer[np.argmax(outer[:, 0])]])
        for s in range(sectors):
            corners = []
            for angle in (theta[s], theta[s + 1]):
                corners.append(
                    np.column_stack([section[:, 0] * np.cos(angle), section[:, 1], section[:, 0] * np.sin(angle)])
                )
            parts.append(np.vstack(corners))
    return parts


def write_collision(path, parts, offset):
    """One named OBJ object per convex piece, with no shared vertices."""
    lines, base, hulls = ["# Convex pieces built from the revolve profile."], 1, []
    for i, part in enumerate(parts):
        hull = trimesh.convex.convex_hull(part + offset)
        hulls.append(hull)
        lines.append(f"o part_{i}")
        lines += [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in hull.vertices]
        lines += [f"f {a + base} {b + base} {c + base}" for a, b, c in hull.faces]
        base += len(hull.vertices)
    path.write_text("\n".join(lines) + "\n")
    return hulls


def build(name, profile, mat, sectors, levels, center_y_split, collision_profile=None):
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    vertices, faces, normals, uv = revolve(profile)
    height = profile[:, 1].max()
    offset = np.array([0.0, -height / 2, 0.0])  # origin at the bounding-box centre

    mesh = trimesh.Trimesh(
        vertices + offset, faces, vertex_normals=normals, process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=mat),
    )
    mesh.export(out / "visual.glb", include_normals=True)

    hulls = write_collision(
        out / "collision_vhacd.obj", convex_parts(profile if collision_profile is None else collision_profile, sectors, levels, center_y_split), offset
    )

    diameter = 2 * profile[:, 0].max()
    metadata = {
        name: {
            # face_idx keeps the footprint in put_basket.load_object_footprint_radius at the true radius.
            "face_idx": 1,
            "rotation_matrix": Y_UP_TO_Z_UP,
            "transform_4x4": [row + [0.0] for row in Y_UP_TO_Z_UP] + [[0.0, 0.0, 0.0, 1.0]],
            "object_half_size": float(height / 2),
            "obb_extents": [float(diameter), float(height), float(diameter)],
        }
    }
    (out / "orientation_result.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Self-checks: a wrong asset stops here rather than reaching the simulator.
    closed = trimesh.Trimesh(vertices, faces, process=False)
    closed.merge_vertices(merge_tex=True, merge_norm=True)
    assert closed.is_watertight, f"{name}: mesh is not watertight"
    assert closed.volume > 0, f"{name}: mesh is inside out"
    assert np.all(profile[:, 0] >= 0), f"{name}: profile crosses the axis"
    collision_points = np.vstack([h.vertices for h in hulls])
    print(
        f"{name}: {len(vertices)} verts, {len(faces)} faces, diameter {diameter * 100:.2f} cm, "
        f"height {height * 100:.2f} cm, volume {closed.volume * 1e6:.1f} cm3, "
        f"{len(hulls)} convex parts, collision extents {np.ptp(collision_points, axis=0).round(4).tolist()}"
    )
    return profile


def main():
    profile = bowl_profile()
    flat_r = profile[np.argmax(profile[:, 1] > 1e-4), 0]
    print(f"bowl flat base diameter: {2 * flat_r * 100:.2f} cm")
    build(
        "bowl", profile, material(BOWL_SRGB, BOWL_ROUGHNESS, seed=1), 20, 4,
        (1e-4, BOWL_COLLISION_WALL + 2e-4), bowl_profile(BOWL_COLLISION_WALL),
    )

    profile = plate_profile()
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(profile, axis=0), axis=1))]
    arc /= arc[-1]
    on_lip = (profile[:, 0] >= PLATE_LIP_R) & (profile[:, 1] >= PLATE_H - 0.0045)
    band = (arc[on_lip].min(), arc[on_lip].max(), PLATE_LIP_ROUGHNESS)
    build("plate", profile, material(PLATE_SRGB, PLATE_ROUGHNESS, band, seed=2), 24, 3, (1e-4, PLATE_FLOOR_Y + 2e-4))


if __name__ == "__main__":
    main()

"""Convert tmp/toys_in_basket/basket.usda into the v2 basket for PutBasket-v1 and EmptyBasket-v1.

Needs usd-core, trimesh, numpy, pillow and scipy. Writes basket/visual.glb, collision_vhacd.obj and
orientation_result.json next to this file, and the same files to ../empty_basket/basket. Meshes are
in metres, Y-up, with the origin at the centre of the bounding box of the visual and collision
together, as in assets/scenes. collision_vhacd.obj isn't made with V-HACD; it keeps that name
because the task loaders expect it.

The USD refers to textures/weave.png and textures/bamboo_rim.png next to it. Those files are used
if they exist. Otherwise this builds visibly labelled stand-ins: pass --textures to point elsewhere.

The USD's own collision cubes aren't used: their 6 mm walls stick out past the rim, dip below the
base and skip the rounded corners. The pieces here follow the visible inner wall and fill out to
the rim's footprint, so the walls are solid and the outside is a straight vertical face.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageFilter
from pxr import Usd, UsdGeom
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
DEFAULT_USD = REPO / "tmp" / "toys_in_basket" / "basket.usda"
CORNER_SECTORS = 3  # convex pieces per rounded corner
SQRT2 = np.sqrt(2.0)


def read_mesh(stage, path):
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    assert np.all(counts == 3), f"{path}: expected triangles"
    assert mesh.GetSubdivisionSchemeAttr().Get() == "none", f"{path}: expected no subdivision"
    assert mesh.GetNormalsInterpolation() == "vertex", f"{path}: expected vertex normals"
    uv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st")
    assert uv.GetInterpolation() == "vertex", f"{path}: expected vertex st"
    world = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
    assert np.allclose(world, np.eye(4)), f"{path}: unexpected transform"
    return {
        "vertices": np.array(mesh.GetPointsAttr().Get(), dtype=float),
        "faces": np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
        "normals": np.array(mesh.GetNormalsAttr().Get(), dtype=float),
        "uv": np.array(uv.Get(), dtype=float),
    }


def wind_to_normals(part):
    """Make each triangle's winding agree with the USD's vertex normals."""
    v, f = part["vertices"], part["faces"]
    face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    flip = np.sum(face_normals * part["normals"][f].sum(1), axis=1) < 0
    f = f.copy()
    f[flip] = f[flip][:, ::-1]
    part["faces"] = f
    return int(flip.sum())


def corner_radius(half, diagonal):
    """Corner radius of a rounded square, from its half width and centre-to-corner distance."""
    return (half * SQRT2 - diagonal) / (SQRT2 - 1.0)


def rounded_square_point(half, radius, stations):
    """Points on a rounded square. Each station is (side index, kind, value)."""
    points = []
    for side, kind, value in stations:
        if kind == "edge":  # value in [-1, 1] along a straight side
            local = np.array([half, value * (half - radius)])
        else:  # value is the angle through the corner, 0 to 90 degrees
            centre = np.array([half - radius, half - radius])
            local = centre + radius * np.array([np.cos(value), np.sin(value)])
        turn = side * np.pi / 2
        c, s = np.cos(turn), np.sin(turn)
        points.append([c * local[0] - s * local[1], s * local[0] + c * local[1]])
    return np.array(points)


def collision_parts(sections, floor):
    """A floor plate, then 4 straight wall blocks and CORNER_SECTORS blocks per corner."""
    half, radius, thickness = floor
    angles = np.linspace(0.0, 2 * np.pi, 49)[:-1]
    # Build the floor outline from the same rounded-square helper, one station per step.
    outline = []
    for side in range(4):
        outline += [(side, "edge", -1.0), (side, "edge", 1.0)]
        outline += [(side, "corner", a) for a in np.linspace(0.0, np.pi / 2, 9)]
    xy = rounded_square_point(half, radius, outline)
    parts = [np.vstack([np.c_[xy, np.zeros(len(xy))], np.c_[xy, np.full(len(xy), thickness)]])]
    del angles

    def block(stations):
        points = []
        for h, r, z in sections:
            xy = rounded_square_point(h, r, stations)
            points.append(np.c_[xy, np.full(len(xy), z)])
        return np.vstack(points)

    for side in range(4):
        parts.append(block([(side, "edge", -1.0), (side, "edge", 1.0)]))
        steps = np.linspace(0.0, np.pi / 2, CORNER_SECTORS + 1)
        for a, b in zip(steps[:-1], steps[1:]):
            parts.append(block([(side, "corner", a), (side, "corner", b)]))
    return parts


def noise(shape, sigma, rng):
    n = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return n / n.std()


def standin_weave(size=1024, strands=6):
    """A seamless over-under rattan weave with small see-through gaps. Not the author's texture."""
    rng = np.random.default_rng(7)
    u, v = np.meshgrid((np.arange(size) + 0.5) / size * strands, (np.arange(size) + 0.5) / size * strands)
    cu, cv = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - cu, v - cv
    horizontal_on_top = (cu + cv) % 2 == 0
    across = np.where(horizontal_on_top, fv, fu)
    along = np.where(horizontal_on_top, fu, fv)
    roll = np.sin(np.pi * np.clip(across, 0, 1)) ** 0.6  # rounded strand profile
    crest = 0.55 + 0.45 * np.sin(np.pi * along)  # strands dip where they pass under
    fibre = noise((size, size), 1.2, rng) * 0.018 + noise((size, size), 22, rng) * 0.03
    shade = np.clip((0.50 + 0.55 * roll * crest + fibre), 0, 1.15)
    tint = np.array([206.0, 164.0, 108.0])
    rgb = np.clip(shade[..., None] * tint, 0, 255)
    gap = (np.minimum(fu, 1 - fu) < 0.035) & (np.minimum(fv, 1 - fv) < 0.035)
    alpha = np.where(gap, 0, 255)
    return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")


def standin_bamboo(size=1024):
    """Lengthwise bamboo fibres with faint nodes. Not the author's texture."""
    rng = np.random.default_rng(11)
    fibres = gaussian_filter(rng.standard_normal((size, size)), (0.8, 60), mode="wrap")
    fibres /= fibres.std()
    u = (np.arange(size) + 0.5) / size
    nodes = np.exp(-(((u * 2) % 1 - 0.5) / 0.012) ** 2)[None, :]
    shade = 0.90 + 0.045 * fibres + 0.03 * noise((size, size), 30, rng) - 0.20 * nodes
    rgb = np.clip(shade[..., None] * np.array([214.0, 176.0, 118.0]), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").filter(ImageFilter.GaussianBlur(0.5))


def texture(directory, name, fallback):
    path = directory / name
    if path.exists():
        print(f"texture: {path}")
        return Image.open(path)
    print(f"texture: {path} is missing, using a stand-in")
    return fallback()


def to_y_up(points, center_z):
    """Z-up to Y-up: world (x, y, z) = (mesh x, -mesh z, mesh y + center_z)."""
    return np.c_[points[:, 0], points[:, 2] - center_z, -points[:, 1]]


def geometry(part, material, center_z, two_sided=False):
    vertices = to_y_up(part["vertices"], center_z)
    normals = to_y_up(part["normals"], 0.0)
    faces, uv = part["faces"], part["uv"]
    if two_sided:  # a separate back face for sheets, since a renderer may cull them
        vertices = np.vstack([vertices, vertices])
        normals = np.vstack([normals, -normals])
        uv = np.vstack([uv, uv])
        faces = np.vstack([faces, faces[:, ::-1] + len(part["vertices"])])
    return trimesh.Trimesh(
        vertices, faces, vertex_normals=normals, process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--textures", type=Path, default=None)
    args = parser.parse_args()

    stage = Usd.Stage.Open(str(args.usd))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0, "expected metersPerUnit = 1"
    assert UsdGeom.GetStageUpAxis(stage) == "Z", "expected upAxis = Z"
    floor = read_mesh(stage, "/Tray/visuals/floor")
    walls = read_mesh(stage, "/Tray/visuals/walls")
    rim = read_mesh(stage, "/Tray/visuals/rim")
    print("faces rewound to match normals:", {n: wind_to_normals(p) for n, p in (("floor", floor), ("walls", walls), ("rim", rim))})

    # Measure every collision dimension from the visual meshes.
    def ring(vertices, z):
        return vertices[np.isclose(vertices[:, 2], z, atol=1e-5)]

    wall_bottom = ring(walls["vertices"], walls["vertices"][:, 2].min())
    bottom_half = np.abs(wall_bottom[:, :2]).max()
    bottom_r = corner_radius(bottom_half, np.linalg.norm(wall_bottom[:, :2], axis=1).max())
    floor_top = float(floor["vertices"][:, 2].max())

    rv = rim["vertices"]
    rim_top_z = float(rv[:, 2].max())
    rim_outer = float(np.abs(rv[:, :2]).max())
    rim_outer_r = corner_radius(rim_outer, np.linalg.norm(rv[:, :2], axis=1).max())
    section = rv[(np.abs(rv[:, 1]) < 1e-3) & (rv[:, 0] > 0)]
    rim_inner = float(section[:, 0].min())
    flank = section[np.isclose(section[:, 0], rim_inner, atol=1e-5)][:, 2]
    top = section[np.isclose(section[:, 2], rim_top_z, atol=1e-5)][:, 0]
    # Sections keep the corner centres fixed relative to the outer rim, as offset curves do.
    r_at = lambda h: max(rim_outer_r - (rim_outer - h), 1e-3)
    sections = [
        (bottom_half, bottom_r, 0.0),
        (bottom_half, bottom_r, floor_top),
        (rim_inner, r_at(rim_inner), float(flank.min())),
        (rim_inner, r_at(rim_inner), float(flank.max())),
        (float(top.min()), r_at(float(top.min())), rim_top_z),
        (float(top.max()), r_at(float(top.max())), rim_top_z),
        (rim_outer, rim_outer_r, float(flank.max())),
        (rim_outer, rim_outer_r, 0.0),
    ]
    print("wall sections (half width, corner radius, z):", np.round(sections, 4).tolist())

    # The base rests on the table at z = 0, 4 mm below the visible floor sheet.
    height = rim_top_z
    center_z = height / 2

    directory = args.textures or args.usd.parent / "textures"
    weave = trimesh.visual.material.PBRMaterial(
        name="Weave",
        baseColorTexture=texture(directory, "weave.png", standin_weave),
        metallicFactor=0.0, roughnessFactor=0.6, alphaMode="MASK", alphaCutoff=0.5, doubleSided=True,
    )
    bamboo = trimesh.visual.material.PBRMaterial(
        name="Rim",
        baseColorTexture=texture(directory, "bamboo_rim.png", standin_bamboo),
        metallicFactor=0.0, roughnessFactor=0.45,
    )
    scene = trimesh.Scene()
    scene.add_geometry(geometry(floor, weave, center_z, two_sided=True), node_name="floor", geom_name="floor")
    scene.add_geometry(geometry(walls, weave, center_z, two_sided=True), node_name="walls", geom_name="walls")
    scene.add_geometry(geometry(rim, bamboo, center_z), node_name="rim", geom_name="rim")

    out = HERE / "basket"
    out.mkdir(parents=True, exist_ok=True)
    scene.export(out / "visual.glb")

    parts = collision_parts(sections, (bottom_half, bottom_r, floor_top))
    lines, base, hulls = ["# Convex pieces measured from basket.usda's visual meshes."], 1, []
    for i, part in enumerate(parts):
        hull = trimesh.convex.convex_hull(to_y_up(part, center_z))
        assert hull.is_volume, f"piece {i} isn't a solid"
        hulls.append(hull)
        lines.append(f"o part_{i}")
        lines += [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in hull.vertices]
        lines += [f"f {a + base} {b + base} {c + base}" for a, b, c in hull.faces]
        base += len(hull.vertices)
    (out / "collision_vhacd.obj").write_text("\n".join(lines) + "\n")

    identity = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    width = 2 * rim_outer
    metadata = {
        "basket": {
            # face_idx keeps put_basket.load_object_footprint_radius near the true half width.
            "face_idx": 1,
            "rotation_matrix": identity.tolist(),
            "transform_4x4": np.block(
                [[identity, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]
            ).tolist(),
            "object_half_size": float(height / 2),
            "obb_extents": [float(width), float(height), float(width)],
        }
    }
    (out / "orientation_result.json").write_text(json.dumps(metadata, indent=2) + "\n")

    target = HERE.parent / "empty_basket" / "basket"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(out, target)

    points = np.vstack([h.vertices for h in hulls])
    print(
        f"basket: {width * 100:.2f} x {width * 100:.2f} x {height * 100:.2f} cm, "
        f"{len(hulls)} convex pieces, collision extents {np.ptp(points, axis=0).round(4).tolist()}"
    )


if __name__ == "__main__":
    main()

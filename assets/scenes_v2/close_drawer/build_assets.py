"""Build the white expanded-metal drawer unit as a ManiSkill articulation, from the photos in tmp/close_drawer.

Overall size: 13.5 in deep (X) x 11.5 in wide (Y) x 22 in tall (Z). Each of the three drawers opens 10 in.
Frame: +X is the front of the unit, Z is up and the origin is on the floor at the centre of the footprint.

Run from any directory with numpy and trimesh available. Writes drawer/ next to this file.
"""
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

INCH = 0.0254
DEPTH, WIDTH, HEIGHT = 13.5 * INCH, 11.5 * INCH, 22 * INCH
TRAVEL = 10 * INCH
OUT = Path(__file__).parent / "drawer"

# Frame, measured from the photos and scaled to the overall size.
LEG_X, LEG_Y = 0.018, 0.012
TOP_H, TOP_BORDER = 0.016, 0.014
CASTER_H = 0.016  # the rear legs stop this far above the floor, on casters

# Drawers.
DRAWER_BOTTOMS = {"top": 0.402, "middle": 0.268, "bottom": 0.134}
DRAWER_H = 0.108
DRAWER_HALF_W = 0.126
DRAWER_LEN = 0.325
DRAWER_FRONT = DEPTH / 2 - 0.007
FLANGE_HALF_W = 0.1305
RUNNER_ABOVE_DRAWER = 0.006
RUNNER_H = 0.015

MESH_COARSE = dict(long=0.0052, short=0.0028, strand=0.0007)
MESH_FINE = dict(long=0.0034, short=0.0020, strand=0.0012)
SHEET_T = 0.0006

WHITE = PBRMaterial(name="white_paint", baseColorFactor=[0.92, 0.92, 0.88, 1.0], roughnessFactor=0.42, metallicFactor=0.0)
RIVET = PBRMaterial(name="rivet_cap", baseColorFactor=[0.90, 0.89, 0.85, 1.0], roughnessFactor=0.5, metallicFactor=0.0)
ZINC = PBRMaterial(name="zinc", baseColorFactor=[0.72, 0.74, 0.76, 1.0], roughnessFactor=0.28, metallicFactor=1.0)
DARK = PBRMaterial(name="caster_rubber", baseColorFactor=[0.14, 0.14, 0.15, 1.0], roughnessFactor=0.6, metallicFactor=0.0)
RECESS = PBRMaterial(name="screw_recess", baseColorFactor=[0.2, 0.2, 0.21, 1.0], roughnessFactor=0.6, metallicFactor=0.3)


class Parts:
    """Triangles collected per material, with hard edges."""

    def __init__(self):
        self.groups = {}

    def add(self, material, vertices, faces):
        self.groups.setdefault(material.name, (material, []))[1].append((np.asarray(vertices, float), np.asarray(faces, int)))

    def add_mesh(self, material, mesh):
        self.add(material, mesh.vertices, mesh.faces)

    def scene(self):
        scene = trimesh.Scene()
        for name, (material, items) in self.groups.items():
            count = 0
            vertices, faces = [], []
            for v, f in items:
                vertices.append(v)
                faces.append(f + count)
                count += len(v)
            mesh = trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), process=False)
            mesh.unmerge_vertices()  # hard edges: every face gets its own normal
            mesh.visual = trimesh.visual.TextureVisuals(material=material)
            scene.add_geometry(mesh, node_name=name, geom_name=name)
        return scene


def chamfered_box(lo, hi, c=0.0008):
    """A box with small bevels, so that its edges catch the light the way painted tube does."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = min(c, *(0.45 * (hi - lo)))
    points = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                sign = np.sign([x - (lo[0] + hi[0]) / 2, y - (lo[1] + hi[1]) / 2, z - (lo[2] + hi[2]) / 2])
                p = np.array([x, y, z])
                for axis in range(3):
                    q = p.copy()
                    q[axis] -= sign[axis] * c
                    points.append(q)
    return trimesh.convex.convex_hull(np.array(points))


def box(parts, lo, hi, material=WHITE, c=0.0008):
    parts.add_mesh(material, chamfered_box(lo, hi, c))


def cylinder(parts, material, start, end, radius, sections=14):
    parts.add_mesh(material, trimesh.creation.cylinder(radius=radius, segment=[start, end], sections=sections))


def _prisms(starts, ends, width_dir, normal, width, thickness):
    """Thin rectangular bars, as vertex and face arrays."""
    half_w = width_dir * width / 2
    half_t = normal * thickness / 2
    corners = []
    for s in (-1, 1):
        for t in (-1, 1):
            corners.append(starts + s * half_w + t * half_t)
            corners.append(ends + s * half_w + t * half_t)
    # vertex order per bar: index = 4 * s_index + 2 * t_index + end_index
    vertices = np.stack(corners, axis=1).reshape(-1, 3)
    base = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    )
    offsets = np.arange(len(starts))[:, None, None] * 8
    return vertices, (base[None] + offsets).reshape(-1, 3)


def expanded_metal(parts, origin, u, v, width, height, long, short, strand, holes=(), material=WHITE):
    """A flat sheet of expanded metal over the rectangle origin + a*u + b*v, long diamond axis along u.

    Each diamond's sides are two families of parallel strands, so the whole sheet takes few triangles.
    `holes` are (a0, a1, b0, b1) rectangles the strands stop at.
    """
    origin, u, v = (np.asarray(x, float) for x in (origin, u, v))
    normal = np.cross(u, v)
    side = np.hypot(long / 2, short / 2)
    spacing = (long * short / 2) / side
    for sign in (1.0, -1.0):
        direction = np.array([long / 2, sign * short / 2]) / side
        perp = np.array([-direction[1], direction[0]])
        corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
        offsets = corners @ perp
        segments = []
        for offset in np.arange(offsets.min() + spacing / 2, offsets.max(), spacing):
            p0 = perp * offset
            t0, t1 = -np.inf, np.inf
            for axis, limit in ((0, width), (1, height)):
                if abs(direction[axis]) < 1e-9:
                    continue
                a, b = sorted(((0 - p0[axis]) / direction[axis], (limit - p0[axis]) / direction[axis]))
                t0, t1 = max(t0, a), min(t1, b)
            if t1 <= t0:
                continue
            spans = [(t0, t1)]
            for a0, a1, b0, b1 in holes:
                ta = sorted(((a0 - p0[0]) / direction[0], (a1 - p0[0]) / direction[0]))
                tb = sorted(((b0 - p0[1]) / direction[1], (b1 - p0[1]) / direction[1]))
                h0, h1 = max(ta[0], tb[0]), min(ta[1], tb[1])
                if h1 <= h0:
                    continue
                next_spans = []
                for s0, s1 in spans:
                    if h0 > s0:
                        next_spans.append((s0, min(h0, s1)))
                    if h1 < s1:
                        next_spans.append((max(h1, s0), s1))
                spans = next_spans
            for s0, s1 in spans:
                segments.append((p0 + direction * s0, p0 + direction * s1))
        if not segments:
            continue
        a = np.array([s[0] for s in segments])
        b = np.array([s[1] for s in segments])
        starts = origin + a[:, 0, None] * u + a[:, 1, None] * v
        ends = origin + b[:, 0, None] * u + b[:, 1, None] * v
        width_dir = perp[0] * u + perp[1] * v
        vertices, faces = _prisms(starts, ends, width_dir, normal, strand, SHEET_T)
        parts.add(material, vertices, faces)


def screw(parts, center, axis):
    """A zinc pan-head Phillips screw whose head faces along `axis`."""
    center, axis = np.asarray(center, float), np.asarray(axis, float)
    cylinder(parts, ZINC, center, center + axis * 0.0022, 0.0052, 20)
    cylinder(parts, ZINC, center + axis * 0.0022, center + axis * 0.0032, 0.0036, 20)
    other = np.array([0.0, 0.0, 1.0])
    across = np.cross(axis, other)
    top = center + axis * 0.00325
    for a, b in ((across, other), (other, across)):
        lo = top + a * -0.0024 + b * -0.0005
        hi = top + a * 0.0024 + b * 0.0005 + axis * 0.0002
        box(parts, np.minimum(lo, hi), np.maximum(lo, hi), RECESS, 0.0)


def build_cabinet():
    parts = Parts()
    collision = []
    hx, hy = DEPTH / 2, WIDTH / 2

    def solid(lo, hi):
        box(parts, lo, hi)
        collision.append((lo, hi))

    # Legs. The front ones reach the floor, the rear ones stand on casters.
    for sx, bottom in ((1, 0.0), (-1, CASTER_H)):
        for sy in (1, -1):
            x0, x1 = sorted((sx * hx, sx * (hx - LEG_X)))
            y0, y1 = sorted((sy * hy, sy * (hy - LEG_Y)))
            box(parts, (x0, y0, bottom), (x1, y1, HEIGHT - TOP_H))
            collision.append(((x0, y0, 0.0), (x1, y1, HEIGHT - TOP_H)))

    # Top: a tube frame around a fine sheet.
    z0, z1 = HEIGHT - TOP_H, HEIGHT
    box(parts, (-hx, -hy, z0), (hx, -hy + TOP_BORDER, z1))
    box(parts, (-hx, hy - TOP_BORDER, z0), (hx, hy, z1))
    box(parts, (-hx, -hy, z0), (-hx + TOP_BORDER, hy, z1))
    box(parts, (hx - TOP_BORDER, -hy, z0), (hx, hy, z1))
    collision.append(((-hx, -hy, z0), (hx, hy, z1)))

    sheet_w, sheet_h = DEPTH - 2 * TOP_BORDER, WIDTH - 2 * TOP_BORDER
    slots = []
    slot_half_len, slot_half_w = 0.122, 0.005
    for y in (-0.075, 0.0, 0.075):
        slots.append((sheet_w / 2 - slot_half_len, sheet_w / 2 + slot_half_len, sheet_h / 2 + y - slot_half_w, sheet_h / 2 + y + slot_half_w))
    origin = (-hx + TOP_BORDER, -hy + TOP_BORDER, HEIGHT - 0.0012)
    expanded_metal(parts, origin, (1, 0, 0), (0, 1, 0), sheet_w, sheet_h, **MESH_FINE, holes=slots)
    for y in (-0.075, 0.0, 0.075):
        slot_origin = (-slot_half_len, y - slot_half_w, HEIGHT - 0.0016)
        expanded_metal(parts, slot_origin, (1, 0, 0), (0, 1, 0), 2 * slot_half_len, 2 * slot_half_w, long=0.02, short=0.0065, strand=0.0008)
        for edge in (y - slot_half_w, y + slot_half_w):
            box(parts, (-slot_half_len, edge - 0.0007, HEIGHT - 0.0016), (slot_half_len, edge + 0.0007, HEIGHT - 0.0008), c=0.0002)

    # Screws through the top frame into the legs, on the left and right faces.
    for sx in (1, -1):
        for sy in (1, -1):
            screw(parts, (sx * (hx - 0.009), sy * hy, HEIGHT - 0.0085), (0, sy, 0))

    # Bottom cross bars, at the front and back only.
    bar_h = 0.020
    solid((hx - LEG_X, -hy + LEG_Y, 0.0), (hx, hy - LEG_Y, bar_h))
    solid((-hx, -hy + LEG_Y, CASTER_H), (-hx + LEG_X, hy - LEG_Y, CASTER_H + bar_h))

    # Casters on the rear legs.
    for sy in (1, -1):
        y = sy * (hy - LEG_Y / 2)
        x = -hx + LEG_X / 2
        cylinder(parts, DARK, (x, y - 0.0045, 0.0085), (x, y + 0.0045, 0.0085), 0.0085, 20)
        for side in (-0.0058, 0.0058):
            box(parts, (x - 0.0055, y + side - 0.0006, 0.004), (x + 0.0055, y + side + 0.0006, CASTER_H), ZINC, 0.0002)
        box(parts, (x - 0.008, y - 0.0065, CASTER_H - 0.0014), (x + 0.008, y + 0.0065, CASTER_H), ZINC, 0.0002)

    # X brace on the back: two flattened rods screwed to the rear legs.
    bx = -hx - 0.0018
    z_high, z_low = 0.444, 0.142
    rod_y = hy - LEG_Y / 2
    for sy, offset in ((1, 0.0), (-1, -0.0032)):
        start = np.array([bx + offset, sy * rod_y, z_high])
        end = np.array([bx + offset, -sy * rod_y, z_low])
        cylinder(parts, WHITE, start, end, 0.002, 10)
        direction = (end - start) / np.linalg.norm(end - start)
        for point in (start, end):
            tip = point + (direction if point is start else -direction) * 0.016
            mid = np.array([(point[0]), (point[1] + tip[1]) / 2, (point[2] + tip[2]) / 2])
            flat = trimesh.creation.box(extents=(0.0012, 0.022, 0.009))
            angle = np.arctan2(direction[2], direction[1])
            flat.apply_transform(trimesh.transformations.rotation_matrix(angle, (1, 0, 0)))
            flat.apply_translation(mid)
            parts.add_mesh(WHITE, flat)
    for sy in (1, -1):
        for z in (z_high, z_low):
            screw(parts, (-hx - 0.002, sy * rod_y, z), (-1, 0, 0))

    # Fixed runners along each side. Their rolled lower lip faces outwards and their front end flares.
    x0, x1 = -hx + LEG_X, hx - LEG_X
    for bottom in DRAWER_BOTTOMS.values():
        top = bottom + DRAWER_H + RUNNER_ABOVE_DRAWER
        for sy in (1, -1):
            y_in, y_out = sy * (FLANGE_HALF_W + 0.0005), sy * (hy - 0.0015)
            y0, y1 = sorted((y_in, y_out))
            box(parts, (x0, y0, top - RUNNER_H), (x1, y1, top - 0.0065))
            box(parts, (x0, y0, top - 0.0065), (x1, y1, top), c=0.0004)
            cylinder(parts, WHITE, (x0, y_out, top - RUNNER_H + 0.004), (x1, y_out, top - RUNNER_H + 0.004), 0.0036, 12)
            collision.append(((x0, y0, top - RUNNER_H), (x1, y1, top)))
            # the flared mouth at the front of the runner
            flare = trimesh.creation.box(extents=(0.03, 0.009, 0.0012))
            flare.apply_transform(trimesh.transformations.rotation_matrix(sy * np.radians(14), (0, 0, 1)))
            flare.apply_translation((x1 - 0.008, sy * (FLANGE_HALF_W + 0.0035), top + 0.0006))
            parts.add_mesh(WHITE, flare)

    return parts, collision


def build_drawer(bottom):
    parts = Parts()
    collision = []
    xf, xr = DRAWER_FRONT, DRAWER_FRONT - DRAWER_LEN
    z0, z1 = bottom, bottom + DRAWER_H
    hw = DRAWER_HALF_W
    t = 0.0045  # depth of the folded frame
    band_top, band_bottom = 0.011, 0.006
    end_stile, stile = 0.011, 0.020

    # The two ends: a folded frame around a coarse sheet.
    for x_out, x_in in ((xf, xf - t), (xr, xr + t)):
        a, b = sorted((x_out, x_in))
        box(parts, (a, -hw, z1 - band_top), (b, hw, z1))
        box(parts, (a, -hw, z0), (b, hw, z0 + band_bottom))
        for sy in (1, -1):
            y0, y1 = sorted((sy * hw, sy * (hw - end_stile)))
            box(parts, (a, y0, z0), (b, y1, z1))
        expanded_metal(
            parts,
            ((a + b) / 2, -hw + end_stile, z0 + band_bottom),
            (0, 1, 0),
            (0, 0, 1),
            2 * (hw - end_stile),
            DRAWER_H - band_top - band_bottom,
            **MESH_COARSE,
        )
        collision.append(((a, -hw, z0), (b, hw, z1)))

    # The two sides.
    body_x0, body_x1 = xr + t, xf - t
    for sy in (1, -1):
        y_out, y_in = sy * hw, sy * (hw - t)
        y0, y1 = sorted((y_out, y_in))
        box(parts, (body_x0, y0, z1 - band_top), (body_x1, y1, z1))
        box(parts, (body_x0, y0, z0), (body_x1, y1, z0 + band_bottom))
        for x_a, x_b in ((body_x1 - stile, body_x1), (body_x0, body_x0 + stile)):
            box(parts, (x_a, y0, z0), (x_b, y1, z1))
            # the raised fold down each corner post
            fold_y0, fold_y1 = sorted((y_out, y_out + sy * 0.0012))
            box(parts, ((x_a + x_b) / 2 - 0.004, fold_y0, z0), ((x_a + x_b) / 2 + 0.004, fold_y1, z1 - band_top), c=0.0003)
        expanded_metal(
            parts,
            (body_x0 + stile, (y_out + y_in) / 2, z0 + band_bottom),
            (1, 0, 0),
            (0, 0, 1),
            body_x1 - body_x0 - 2 * stile,
            DRAWER_H - band_top - band_bottom,
            **MESH_COARSE,
        )
        # the shallow pressed rectangle across each side sheet
        for z in (z0 + 0.030,):
            ry0, ry1 = sorted(((y_out + y_in) / 2, (y_out + y_in) / 2 + sy * 0.0007))
            box(parts, (body_x0 + 0.06, ry0, z), (body_x1 - 0.06, ry1, z + 0.0012), c=0.0002)
        # the galvanised-looking hanger at the top of each side
        hy0, hy1 = sorted((sy * hw, sy * FLANGE_HALF_W))
        box(parts, (xr, hy0, z1 - 0.0016), (xf, hy1, z1), c=0.0003)
        collision.append(((body_x0, y0, z0), (body_x1, y1, z1)))
        collision.append(((xr, hy0, z1 - 0.0016), (xf, hy1, z1)))
        # white stop buttons on the outside of both corner posts
        for x in (body_x1 - stile / 2, body_x0 + stile / 2):
            y_a, y_b = sy * hw, sy * (hw + 0.003)
            cylinder(parts, RIVET, (x, y_a, z1 - 0.007), (x, y_b, z1 - 0.007), 0.0056, 18)

        # a slotted strip along the bottom of the inside of each side
        strip_y = sy * (hw - t - 0.0008)
        sy0, sy1 = sorted((strip_y, strip_y + sy * 0.0012))
        strip_z0, strip_z1 = z0 + 0.0035, z0 + 0.0215
        slot_z0, slot_z1 = strip_z0 + 0.0065, strip_z1 - 0.0065
        strip_x0, strip_x1 = body_x0 + stile, body_x1 - stile
        box(parts, (strip_x0, sy0, strip_z0), (strip_x1, sy1, slot_z0), c=0.0002)
        box(parts, (strip_x0, sy0, slot_z1), (strip_x1, sy1, strip_z1), c=0.0002)
        pitch, slot_len = 0.0215, 0.0145
        x = strip_x0
        while x < strip_x1:
            post_end = min(x + pitch - slot_len, strip_x1)
            box(parts, (x, sy0, slot_z0), (post_end, sy1, slot_z1), c=0.0001)
            x += pitch

    # The floor: a coarse sheet with a flat strip and rivet along each end.
    floor_z = z0 + 0.0028
    strip = 0.03
    floor_x0, floor_x1 = body_x0 + strip, body_x1 - strip
    expanded_metal(parts, (floor_x0, -hw + t, floor_z), (1, 0, 0), (0, 1, 0), floor_x1 - floor_x0, 2 * (hw - t), **MESH_COARSE)
    for a, b in ((body_x0, floor_x0), (floor_x1, body_x1)):
        box(parts, (a, -hw + t, floor_z - 0.0008), (b, hw - t, floor_z + 0.0008), c=0.0002)
        for sy in (1, -1):
            cylinder(parts, RIVET, ((a + b) / 2, sy * (hw - 0.04), floor_z), ((a + b) / 2, sy * (hw - 0.04), floor_z) + np.array([0, 0, 0.0028]), 0.004, 14)
    # the shallow pressed rectangle across the floor sheet
    for y in (-0.085, 0.085):
        box(parts, (floor_x0 + 0.03, y - 0.0006, floor_z), (floor_x1 - 0.03, y + 0.0006, floor_z + 0.0007), c=0.0002)
    for x in (floor_x0 + 0.03, floor_x1 - 0.03):
        box(parts, (x - 0.0006, -0.085, floor_z), (x + 0.0006, 0.085, floor_z + 0.0007), c=0.0002)
    collision.append(((body_x0, -hw + t, z0), (body_x1, hw - t, floor_z + 0.0008)))

    return parts, collision


def box_inertial(collision, mass):
    """Mass, centre of mass and diagonal inertia, spreading the mass over the boxes by volume."""
    boxes = [(np.asarray(lo, float), np.asarray(hi, float)) for lo, hi in collision]
    volumes = np.array([np.prod(hi - lo) for lo, hi in boxes])
    masses = mass * volumes / volumes.sum()
    com = sum(m * (lo + hi) / 2 for m, (lo, hi) in zip(masses, boxes)) / mass
    inertia = np.zeros(3)
    for m, (lo, hi) in zip(masses, boxes):
        e = hi - lo
        d = (lo + hi) / 2 - com
        inertia += m / 12 * np.array([e[1] ** 2 + e[2] ** 2, e[0] ** 2 + e[2] ** 2, e[0] ** 2 + e[1] ** 2])
        inertia += m * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    return com, inertia


def link_xml(name, mesh_name, collision, mass):
    com, inertia = box_inertial(collision, mass)
    lines = [
        f'  <link name="{name}">',
        "    <inertial>",
        f'      <origin xyz="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}"/>',
        f'      <mass value="{mass}"/>',
        f'      <inertia ixx="{inertia[0]:.6f}" ixy="0" ixz="0" iyy="{inertia[1]:.6f}" iyz="0" izz="{inertia[2]:.6f}"/>',
        "    </inertial>",
        f'    <visual><geometry><mesh filename="{mesh_name}"/></geometry></visual>',
    ]
    for lo, hi in collision:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c, e = (lo + hi) / 2, hi - lo
        lines.append(
            f'    <collision><origin xyz="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}"/>'
            f'<geometry><box size="{e[0]:.5f} {e[1]:.5f} {e[2]:.5f}"/></geometry></collision>'
        )
    lines.append("  </link>")
    return lines


def main():
    OUT.mkdir(exist_ok=True)
    urdf = ['<?xml version="1.0"?>', '<robot name="mesh_drawer_unit">']
    total = 0

    parts, collision = build_cabinet()
    scene = parts.scene()
    scene.export(OUT / "cabinet.glb")
    total += sum(len(g.faces) for g in scene.geometry.values())
    urdf += link_xml("cabinet", "cabinet.glb", collision, 4.5)
    all_bounds = [scene.bounds]

    for name, bottom in DRAWER_BOTTOMS.items():
        parts, collision = build_drawer(bottom)
        scene = parts.scene()
        scene.export(OUT / f"drawer_{name}.glb")
        total += sum(len(g.faces) for g in scene.geometry.values())
        all_bounds.append(scene.bounds)
        urdf += link_xml(f"drawer_{name}", f"drawer_{name}.glb", collision, 0.9)
        urdf += [
            f'  <joint name="drawer_{name}_joint" type="prismatic">',
            '    <parent link="cabinet"/>',
            f'    <child link="drawer_{name}"/>',
            '    <origin xyz="0 0 0"/>',
            '    <axis xyz="1 0 0"/>',
            f'    <limit lower="0" upper="{TRAVEL:.4f}" effort="100" velocity="1"/>',
            '    <dynamics damping="4" friction="0.5"/>',
            "  </joint>",
        ]
    urdf.append("</robot>")
    (OUT / "mobility.urdf").write_text("\n".join(urdf) + "\n")

    lo = np.min([b[0] for b in all_bounds], axis=0)
    hi = np.max([b[1] for b in all_bounds], axis=0)
    print("triangles:", total)
    print("visual bounds (cm):", (lo * 100).round(2).tolist(), (hi * 100).round(2).tolist())
    print("wanted (cm): depth", round(DEPTH * 100, 2), "width", round(WIDTH * 100, 2), "height", round(HEIGHT * 100, 2), "travel", round(TRAVEL * 100, 2))


if __name__ == "__main__":
    main()

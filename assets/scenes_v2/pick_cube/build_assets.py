"""Writes the 4 cm blue cube used by PickCube-v2 and PushCube-v2. Needs only the standard library."""
import json
import struct
from pathlib import Path

HALF_SIZE = 0.02  # m, the size both tasks have always used
BASE_COLOR = [12 / 255, 42 / 255, 160 / 255, 1.0]  # the blue PushCube has always used
ROOT = Path(__file__).resolve().parent.parent
TASKS = ("pick_cube", "push_cube")

FACES = [  # normal, then two edge directions whose cross product is the normal
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
]


def corner(normal, u, v, su, sv):
    return [HALF_SIZE * (normal[i] + su * u[i] + sv * v[i]) for i in range(3)]


def glb_bytes():
    # Each face has its own four vertices, so the edges stay sharp.
    positions, normals, indices = [], [], []
    for normal, u, v in FACES:
        start = len(positions)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            positions.append(corner(normal, u, v, su, sv))
            normals.append(list(normal))
        indices += [start, start + 1, start + 2, start, start + 2, start + 3]

    pos_bin = b"".join(struct.pack("<3f", *p) for p in positions)
    nor_bin = b"".join(struct.pack("<3f", *n) for n in normals)
    idx_bin = b"".join(struct.pack("<H", i) for i in indices)
    idx_bin += b"\0" * (-len(idx_bin) % 4)
    binary = pos_bin + nor_bin + idx_bin

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "cube"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"name": "blue", "pbrMetallicRoughness": {"baseColorFactor": BASE_COLOR, "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bin), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bin), "byteLength": len(nor_bin), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bin) + len(nor_bin), "byteLength": len(indices) * 2, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": [-HALF_SIZE] * 3, "max": [HALF_SIZE] * 3},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
    }
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    return (
        struct.pack("<3I", 0x46546C67, 2, total)
        + struct.pack("<2I", len(text), 0x4E4F534A) + text
        + struct.pack("<2I", len(binary), 0x004E4942) + binary
    )


def collision_obj():
    lines = ["o cube"]
    signs = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    lines += [f"v {x * HALF_SIZE} {y * HALF_SIZE} {z * HALF_SIZE}" for x, y, z in signs]
    for a, b, c, d in ((1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)):
        lines += [f"f {a} {b} {c}", f"f {a} {c} {d}"]
    return "\n".join(lines) + "\n"


for task in TASKS:
    folder = ROOT / task / "cube"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "visual.glb").write_bytes(glb_bytes())
    (folder / "collision_vhacd.obj").write_text(collision_obj())
    print("wrote", folder)

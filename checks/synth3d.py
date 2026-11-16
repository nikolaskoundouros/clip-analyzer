"""A test clip with KNOWN 3D camera moves, to check the scanner's labels.

A layered scene (a far wall, a subject in the middle, two near objects) is filmed by a virtual
pinhole camera that does, 30 frames each: dolly in, dolly out, truck right, truck left, orbit right,
orbit left, lens zoom in, lens zoom out, pan right, pan left. Every move returns to the start.
Writes OUT/moves3d.mp4 and OUT/truth_moves3d.csv (t, label).
Usage: python3 synth3d.py FAR.jpg MID.jpg NEAR.jpg OUT_DIR
"""
import csv
import math
import sys

import cv2
import numpy as np

far_p, mid_p, near_p, out = sys.argv[1:5]
W, H, FPS, F0 = 540, 960, 30, 700.0


def oval(img, w):
    img = cv2.resize(img, (w, int(w * img.shape[0] / img.shape[1])))
    a = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(a, (img.shape[1] // 2, img.shape[0] // 2), (img.shape[1] // 2 - 4, img.shape[0] // 2 - 4), 0, 0, 360, 1.0, -1)
    return img, cv2.GaussianBlur(a, (9, 9), 0)


far = cv2.imread(far_p)
mid, mid_a = oval(cv2.imread(mid_p)[300:1500, 140:940], 400)
near, near_a = oval(cv2.imread(near_p)[200:1100, 200:880], 300)
# (texture, alpha or None, world units per texture pixel, centre x, centre y, depth z)
LAYERS = [(far, None, 0.0135, 0.0, 0.0, 10.0),
          (mid, mid_a, 0.0050, 0.0, 0.4, 5.0),
          (near, near_a, 0.0022, -0.85, 1.1, 3.0),
          (near, near_a, 0.0022, 0.9, -1.2, 3.0)]


def frame(C, yaw, f):
    """C = camera position, yaw = turn to the right in radians, f = focal length in pixels."""
    R = np.array([[math.cos(yaw), 0, -math.sin(yaw)], [0, 1, 0], [math.sin(yaw), 0, math.cos(yaw)]])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    img = np.zeros((H, W, 3), np.float32)
    for tex, alpha, s, x0, y0, z in LAYERS:
        th, tw = tex.shape[:2]
        origin = np.array([x0 - tw / 2 * s, y0 - th / 2 * s, z]) - C
        Hm = K @ np.column_stack([R[:, 0] * s, R[:, 1] * s, R @ origin])
        col = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT if alpha is None else cv2.BORDER_CONSTANT).astype(np.float32)
        if alpha is None:
            img = col
        else:
            al = cv2.warpPerspective(alpha, Hm, (W, H), flags=cv2.INTER_LINEAR)[:, :, None]
            img = img * (1 - al) + col * al
    return np.clip(img, 0, 255).astype(np.uint8)


N = 30
moves = []
for name, fn in [
    ("dolly in", lambda u: (np.array([0, 0, 1.6 * u]), 0.0, F0)),
    ("dolly out", lambda u: (np.array([0, 0, 1.6 * (1 - u)]), 0.0, F0)),
    ("truck right", lambda u: (np.array([1.2 * u, 0, 0]), 0.0, F0)),
    ("truck left", lambda u: (np.array([1.2 * (1 - u), 0, 0]), 0.0, F0)),
    ("orbit right", lambda u: (np.array([5 * math.sin(0.35 * u), 0, 5 - 5 * math.cos(0.35 * u)]), -0.35 * u, F0)),
    ("orbit left", lambda u: (np.array([5 * math.sin(0.35 * (1 - u)), 0, 5 - 5 * math.cos(0.35 * (1 - u))]), -0.35 * (1 - u), F0)),
    ("zoom in", lambda u: (np.zeros(3), 0.0, F0 * (1 + 0.45 * u))),
    ("zoom out", lambda u: (np.zeros(3), 0.0, F0 * (1 + 0.45 * (1 - u)))),
    ("pan right", lambda u: (np.zeros(3), 0.20 * u, F0)),
    ("pan left", lambda u: (np.zeros(3), 0.20 * (1 - u), F0)),
]:
    moves.append((name, fn))
vw = cv2.VideoWriter(f"{out}/moves3d.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
rows, k = [], 0
for name, fn in moves:
    for i in range(N):
        C, yaw, f = fn(i / (N - 1))
        vw.write(frame(C, yaw, f))
        rows.append(dict(t=round(k / FPS, 4), label=name))
        k += 1
vw.release()
with open(f"{out}/truth_moves3d.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["t", "label"])
    w.writeheader()
    w.writerows(rows)
print("moves3d", k, "frames")

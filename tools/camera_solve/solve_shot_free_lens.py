"""Solve the reference's opening shot (frames 0-54, 0 to 1.8 s) with COLMAP (BSD), then read the camera path.

Usage: python solve.py DIR   (DIR has images/ and masks/ ; writes DIR/db.db, DIR/sparse/, DIR/path.json)
Reports: frames solved, mean reprojection error (px), and for the orbit part (frames 20-54) the angle the
camera travelled around the point it looks at, its rise, and its change in distance.
"""
import json
import os
import sys

import numpy as np
import pycolmap

D = sys.argv[1]
db, imgs, out = f"{D}/db.db", f"{D}/images", f"{D}/sparse"
os.makedirs(out, exist_ok=True)
if os.path.exists(db):
    os.remove(db)
reader = pycolmap.ImageReaderOptions()
reader.mask_path = f"{D}/masks"
reader.camera_model = "SIMPLE_RADIAL"
pycolmap.extract_features(db, imgs, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
pairing = pycolmap.SequentialPairingOptions()
pairing.overlap = 12
pycolmap.match_sequential(db, pairing_options=pairing)
recs = pycolmap.incremental_mapping(db, imgs, out)
print("models:", {k: r.num_reg_images() for k, r in recs.items()})
if not recs:
    sys.exit("NO MODEL")
rec = max(recs.values(), key=lambda r: r.num_reg_images())
print(rec.summary())

path = {}
for im in rec.images.values():
    if not im.has_pose:
        continue
    pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
    Rm = pose.rotation.matrix()
    C = -Rm.T @ pose.translation
    path[int(im.name[1:5])] = dict(center=C.tolist(), R=Rm.tolist())
frames = sorted(path)
print("solved frames:", len(frames), "of 55:", frames[0], "...", frames[-1], "missing:", [f for f in range(55) if f not in path])

# orbit part: the point all the views look at (closest point to every viewing ray)
sel = [f for f in frames if f >= 20]
A, b = np.zeros((3, 3)), np.zeros(3)
for f in sel:
    R, C = np.array(path[f]["R"]), np.array(path[f]["center"])
    d = R.T @ np.array([0, 0, 1.0])
    P = np.eye(3) - np.outer(d, d)
    A += P
    b += P @ C
target = np.linalg.solve(A, b)
up = -np.mean([np.array(path[f]["R"]).T @ np.array([0, 1.0, 0]) for f in sel], axis=0)
up /= np.linalg.norm(up)
flat = lambda v: v - up * (v @ up)
c0, c1 = np.array(path[sel[0]]["center"]) - target, np.array(path[sel[-1]]["center"]) - target
ang = np.degrees(np.arctan2(np.cross(flat(c0), flat(c1)) @ up, flat(c0) @ flat(c1)))
rays = [np.array(path[f]["center"]) - target for f in sel]
steps = [np.degrees(np.arctan2(np.cross(flat(a), flat(b2)) @ up, flat(a) @ flat(b2))) for a, b2 in zip(rays[:-1], rays[1:])]
summary = dict(solved=len(frames), mean_reproj_px=round(float(rec.compute_mean_reprojection_error()), 3),
               orbit_frames=[sel[0], sel[-1]], orbit_deg=round(float(ang), 1), orbit_deg_summed=round(float(np.sum(steps)), 1),
               rise_over_distance=round(float((c1 - c0) @ up / np.linalg.norm(c0)), 3),
               distance_ratio=round(float(np.linalg.norm(c1) / np.linalg.norm(c0)), 3))
print("SUMMARY", json.dumps(summary))
json.dump(dict(summary=summary, target=target.tolist(), up=up.tolist(), path=path), open(f"{D}/path.json", "w"))

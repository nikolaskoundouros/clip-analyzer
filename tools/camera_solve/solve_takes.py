"""Solve whole TAKES of the reference (not shots): the camera path through whips and blur, with COLMAP (BSD).

Why (15 Nov 2026): solve_all_shots.py solved each hand-cut "shot" alone, so a whip (blurred frames) split the
path and some shots solved 2 of 26 frames (S9) or 0 (S2, S4). A take is the stretch between two real cuts, so it is
one continuous camera path. If the take really has a jump cut inside, COLMAP gives two models: every model is kept.

Usage: python solve_takes.py WORK FIRST LAST [NAME]
  WORK/all/fNNNN.png  frames at 540x960 (as made by solve_all_shots.py)
  Writes WORK/<NAME>/models.json: for every model: frames solved, mean reprojection error, per-frame camera
  centre + rotation, and the frames of the take that no model registered (the whip / blur frames).
Settings that differ from solve_all_shots.py (all env, so a run can be repeated exactly):
  FEATURES (default 8192) more features per frame; DSP=1 domain-size pooling (more stable on blur);
  OVERLAP (default 20) frames matched ahead; POSE_INL (default 12) and POSE_RATIO (default 0.12): how little a
  blurred frame needs to register (COLMAP defaults 30 and 0.25)."""
import json
import os
import shutil
import sys

import cv2
import numpy as np
import pycolmap

WORK, A, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
NAME = sys.argv[4] if len(sys.argv) > 4 else f"take_{A}_{B}"
env = lambda k, d: type(d)(os.environ.get(k, d))
FOCAL = "961.431,270,480,0.1795"          # the opening shot's solved lens at 540x960 (short moves cannot solve their own)
D = os.path.join(WORK, NAME)
shutil.rmtree(D, ignore_errors=True)
os.makedirs(f"{D}/images")
os.makedirs(f"{D}/masks")
for f in range(A, B + 1):
    shutil.copy(f"{WORK}/all/f{f:04d}.png", f"{D}/images/f{f:04d}.png")
    m = np.full((960, 540), 255, np.uint8)
    m[int(0.70 * 960):int(0.80 * 960), :] = 0          # the caption strip
    cv2.imwrite(f"{D}/masks/f{f:04d}.png.png", m)
db = f"{D}/db.db"
reader = pycolmap.ImageReaderOptions()
reader.mask_path = f"{D}/masks"
reader.camera_model = "SIMPLE_RADIAL"
reader.camera_params = FOCAL
fe = pycolmap.FeatureExtractionOptions()
fe.sift.max_num_features = env("FEATURES", 8192)
fe.sift.domain_size_pooling = bool(env("DSP", 1))
fe.sift.estimate_affine_shape = bool(env("DSP", 1))
pycolmap.extract_features(db, f"{D}/images", camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=fe)
pairing = pycolmap.SequentialPairingOptions()
pairing.overlap = env("OVERLAP", 20)
pycolmap.match_sequential(db, pairing_options=pairing)
opt = pycolmap.IncrementalPipelineOptions()
opt.ba_refine_focal_length = False
opt.ba_refine_extra_params = False
opt.min_num_matches = 10
opt.mapper.init_min_tri_angle = 2.0
opt.mapper.init_min_num_inliers = 50
opt.mapper.abs_pose_min_num_inliers = env("POSE_INL", 12)
opt.mapper.abs_pose_min_inlier_ratio = env("POSE_RATIO", 0.12)
os.makedirs(f"{D}/sparse", exist_ok=True)
recs = pycolmap.incremental_mapping(db, f"{D}/images", f"{D}/sparse", options=opt)
out = dict(take=[A, B], settings=dict(features=env("FEATURES", 8192), dsp=env("DSP", 1), overlap=env("OVERLAP", 20),
                                      pose_inliers=env("POSE_INL", 12), pose_ratio=env("POSE_RATIO", 0.12)), models=[])
seen = set()
for k, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
    path = {}
    for im in rec.images.values():
        if im.has_pose:
            pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            R = pose.rotation.matrix()
            path[int(im.name[1:5])] = dict(center=(-R.T @ pose.translation).tolist(), R=R.tolist())
    seen |= set(path)
    fr = sorted(path)
    err = float(np.mean([p.error for p in rec.points3D.values()])) if rec.num_points3D() else None
    out["models"].append(dict(model=k, frames=len(fr), first=fr[0] if fr else None, last=fr[-1] if fr else None,
                              points=rec.num_points3D(), mean_reproj_px=err, path=path))
out["unregistered"] = [f for f in range(A, B + 1) if f not in seen]
json.dump(out, open(f"{D}/models.json", "w"))
print(f"TAKE {A}-{B}: {len(out['models'])} models, " +
      ", ".join(f"[{m['first']}-{m['last']}: {m['frames']} frames, {m['mean_reproj_px']:.2f} px]" for m in out["models"]) +
      f"; unregistered {len(out['unregistered'])} of {B - A + 1}")

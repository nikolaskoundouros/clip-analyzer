"""Solve every shot of the reference from 2 to 10 s with COLMAP (BSD), one model per shot.

Usage: python solve_shots.py WORK VIDEO
For each shot (frame ranges from runs/021 build.py's shot starts): frames at 540x960 with the caption band
masked, then COLMAP with the lens fixed at the opening shot's solved value (961.4 px focal; short shots
cannot work out a lens on their own) and a small starting angle allowed (short moves). Writes
WORK/<shot>/path.json: solved cameras, the scene's up, the anchor (solved points in the middle of the
middle frame: the shot's subject) and quality numbers. A shot that does not solve is reported, not faked.
"""
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pycolmap

WORK, VIDEO = sys.argv[1], sys.argv[2]
SHOTS = {"S2": (60, 68), "S3": (69, 96), "S4": (97, 101), "S5": (102, 119), "S6": (120, 166), "S7": (167, 175),
         "S8": (176, 199), "S9": (200, 225), "S10": (226, 280), "S11": (281, 302)}
FOCAL = "961.431,270,480,0.1795"

allf = os.path.join(WORK, "all")
os.makedirs(allf, exist_ok=True)
if not os.listdir(allf):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", VIDEO, "-vf", "select=lt(n\\,303),scale=540:960", "-vsync", "0",
                    "-start_number", "0", os.path.join(allf, "f%04d.png")], check=True)
report = {}
for shot, (a, b) in SHOTS.items():
    D = os.path.join(WORK, shot)
    shutil.rmtree(D, ignore_errors=True)
    os.makedirs(f"{D}/images")
    os.makedirs(f"{D}/masks")
    for f in range(a, b + 1):
        shutil.copy(f"{allf}/f{f:04d}.png", f"{D}/images/f{f:04d}.png")
        m = np.full((960, 540), 255, np.uint8)
        m[int(0.70 * 960):int(0.80 * 960), :] = 0
        cv2.imwrite(f"{D}/masks/f{f:04d}.png.png", m)
    db = f"{D}/db.db"
    reader = pycolmap.ImageReaderOptions()
    reader.mask_path = f"{D}/masks"
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = FOCAL
    pycolmap.extract_features(db, f"{D}/images", camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 12
    pycolmap.match_sequential(db, pairing_options=pairing)
    opt = pycolmap.IncrementalPipelineOptions()
    opt.ba_refine_focal_length = False
    opt.ba_refine_extra_params = False
    opt.min_num_matches = 15
    opt.mapper.init_min_tri_angle = 2.0
    opt.mapper.init_min_num_inliers = 50
    os.makedirs(f"{D}/sparse", exist_ok=True)
    try:
        recs = pycolmap.incremental_mapping(db, f"{D}/images", f"{D}/sparse", options=opt)
    except Exception as e:
        recs = {}
        print(shot, "mapping error", e)
    n = b - a + 1
    if not recs:
        report[shot] = dict(frames=n, solved=0)
        print(f"{shot} frames {a}-{b}: NOT SOLVED")
        continue
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    path = {}
    for im in rec.images.values():
        if im.has_pose:
            pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            R = pose.rotation.matrix()
            path[int(im.name[1:5])] = dict(center=(-R.T @ pose.translation).tolist(), R=R.tolist())
    fr = sorted(path)
    up = -np.mean([np.array(path[f]["R"]).T @ np.array([0, 1.0, 0]) for f in fr], axis=0)
    up /= np.linalg.norm(up)
    mid = fr[len(fr) // 2]
    im = next(i for i in rec.images.values() if i.name == f"f{mid:04d}.png")
    pts = [rec.points3D[p.point3D_id].xyz for p in im.points2D
           if p.has_point3D() and 0.30 * 540 <= p.xy[0] <= 0.70 * 540 and 0.25 * 960 <= p.xy[1] <= 0.70 * 960]
    anchor = np.median(np.array(pts), axis=0) if len(pts) >= 5 else None
    turn = max(np.degrees(np.arccos(np.clip((np.trace(np.array(path[f]["R"]) @ np.array(path[fr[0]]["R"]).T) - 1) / 2, -1, 1))) for f in fr)
    travel = max(np.linalg.norm(np.array(path[f]["center"]) - np.array(path[fr[0]]["center"])) for f in fr)
    dist0 = float(np.linalg.norm(np.array(path[fr[0]]["center"]) - anchor)) if anchor is not None else None
    q = dict(frames=n, solved=len(fr), first=fr[0], last=fr[-1], reproj_px=round(float(rec.compute_mean_reprojection_error()), 3),
             points=rec.num_points3D(), anchor_points=len(pts), max_turn_deg=round(float(turn), 1),
             travel_over_first_distance=round(float(travel / dist0), 3) if dist0 else None)
    report[shot] = q
    json.dump(dict(shot=shot, range=[a, b], quality=q, up=up.tolist(), anchor=anchor.tolist() if anchor is not None else None,
                   path=path), open(f"{D}/path.json", "w"))
    print(f"{shot} frames {a}-{b}: {json.dumps(q)}", flush=True)
json.dump(report, open(os.path.join(WORK, "report.json"), "w"), indent=1)
print("DONE")

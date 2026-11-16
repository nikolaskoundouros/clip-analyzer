"""Solve the reference's 1.9-4.1 s (frames 57-121) with COLMAP, people and captions masked, lens fixed
at the opening shot's solved value; then measure the camera's turn and its path round the point all views look at."""
import json, math, os, sys
import numpy as np, pycolmap
D = sys.argv[1]
db = f"{D}/db.db"
if os.path.exists(db):
    os.remove(db)
r = pycolmap.ImageReaderOptions(); r.mask_path = f"{D}/masks"; r.camera_model = "SIMPLE_RADIAL"; r.camera_params = "961.431,270,480,0.1795"
pycolmap.extract_features(db, f"{D}/images", camera_mode=pycolmap.CameraMode.SINGLE, reader_options=r)
pr = pycolmap.SequentialPairingOptions(); pr.overlap = 15
pycolmap.match_sequential(db, pairing_options=pr)
o = pycolmap.IncrementalPipelineOptions(); o.ba_refine_focal_length = False; o.ba_refine_extra_params = False
o.min_num_matches = 12; o.mapper.init_min_tri_angle = 2.0; o.mapper.init_min_num_inliers = 40
os.makedirs(f"{D}/sparse", exist_ok=True)
recs = pycolmap.incremental_mapping(db, f"{D}/images", f"{D}/sparse", options=o)
print("models:", {k: v.num_reg_images() for k, v in recs.items()})
out = {}
for k, rec in recs.items():
    P = {}
    for im in rec.images.values():
        if im.has_pose:
            pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            R = pose.rotation.matrix(); P[int(im.name[1:5])] = dict(center=(-R.T @ pose.translation).tolist(), R=R.tolist())
    fr = sorted(P)
    up = -np.mean([np.array(P[f]["R"]).T @ np.array([0, 1.0, 0]) for f in fr], axis=0); up /= np.linalg.norm(up)
    fwd = [np.array(P[f]["R"]).T @ np.array([0, 0, 1.0]) for f in fr]
    flat = lambda v: v - up * (v @ up)
    yaw = [0.0]
    for a, b in zip(fwd[:-1], fwd[1:]):
        yaw.append(yaw[-1] + math.degrees(math.atan2(np.cross(flat(a), flat(b)) @ up, flat(a) @ flat(b))))
    A, bb = np.zeros((3, 3)), np.zeros(3)
    for f, dv in zip(fr, fwd):
        Pm = np.eye(3) - np.outer(dv, dv); A += Pm; bb += Pm @ np.array(P[f]["center"])
    c = np.linalg.solve(A, bb)
    C = [flat(np.array(P[f]["center"]) - c) for f in fr]
    orb = [0.0]
    for a, b in zip(C[:-1], C[1:]):
        orb.append(orb[-1] + math.degrees(math.atan2(np.cross(a, b) @ up, a @ b)))
    dist = [float(np.linalg.norm(np.array(P[f]["center"]) - c)) for f in fr]
    hgt = [float((np.array(P[f]["center"]) - c) @ up) for f in fr]
    q = dict(frames=[fr[0], fr[-1]], solved=len(fr), reproj=round(float(rec.compute_mean_reprojection_error()), 3),
             view_turn_deg=round(yaw[-1], 1), round_centre_deg=round(orb[-1], 1), dist_first_last=[round(dist[0], 3), round(dist[-1], 3)],
             height_change_over_first_dist=round((hgt[-1] - hgt[0]) / dist[0], 2))
    print("model", k, json.dumps(q))
    print("  view turn every 4 frames:", [(fr[i], round(yaw[i], 1)) for i in range(0, len(fr), 4)])
    out[k] = dict(q=q, frames=fr, yaw=yaw, orbit=orb, dist=dist, height=hgt, up=up.tolist(), centre=c.tolist(), path=P)
json.dump({str(k): v for k, v in out.items()}, open(f"{D}/orbit.json", "w"))

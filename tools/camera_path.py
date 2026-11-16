"""Whip bridge: one camera pose for EVERY frame of a take, from COLMAP poses on the sharp frames plus OpenCV
for the blurred frames in between (15 Nov 2026).

COLMAP registers sharp frames only. A whip is mostly the camera turning on the spot, and a turn can be measured
without 3D points: for a pure rotation the picture moves by the homography H = K R K^-1, so R = K^-1 H K.
Bridge(a, b): the frames a and b have COLMAP poses, the frames between do not.
  1 frame-to-frame turn between neighbours (SIFT + RANSAC homography; fallback ECC on a small copy for heavy blur)
  2 chain the turns from a; at b the chain misses COLMAP's pose by an error rotation E; spread E evenly over the
    gap (a fraction k/n of it at step k), so the bridge starts and ends exactly on COLMAP's poses
  3 camera centre: smooth interpolation between the two solved centres (a whip is mostly turning, not travelling)
Poses use COLMAP's convention: R = world to camera, centre = -R^T t.
"""
import json
import math

import cv2
import numpy as np

W, H = 540, 960
FOCAL = 961.431
K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1.0]])
KI = np.linalg.inv(K)
CAPTION = (int(0.70 * H), int(0.80 * H))               # rows to ignore (burned-in captions)


def _gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = g.copy()
    g[CAPTION[0]:CAPTION[1], :] = 0
    return g


def _nearest_rotation(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R


def turn_between(img_a, img_b):
    """Rotation taking camera-a coordinates to camera-b coordinates (x_b = R x_a), the method used, and a quality number.
    Returns (None, 'fail', 0) when nothing can be measured."""
    ga, gb = _gray(img_a), _gray(img_b)
    mask = np.full(ga.shape, 255, np.uint8)
    mask[CAPTION[0]:CAPTION[1], :] = 0
    sift = cv2.SIFT_create(nfeatures=4000)
    ka, da = sift.detectAndCompute(ga, mask)
    kb, db = sift.detectAndCompute(gb, mask)
    if da is not None and db is not None and len(ka) > 20 and len(kb) > 20:
        m = cv2.BFMatcher().knnMatch(da, db, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
        if len(good) >= 12:
            pa = np.float32([ka[g.queryIdx].pt for g in good])
            pb = np.float32([kb[g.trainIdx].pt for g in good])
            Hm, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
            if Hm is not None and inl.sum() >= 12:
                return _nearest_rotation(KI @ Hm @ K), "sift", int(inl.sum())
    s = 4                                                      # heavy blur: ECC on a quarter-size copy
    sa, sb = cv2.resize(ga, (W // s, H // s)), cv2.resize(gb, (W // s, H // s))
    warp = np.eye(3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(sa, sb, warp, cv2.MOTION_HOMOGRAPHY,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), None, 5)
    except cv2.error:
        return None, "fail", 0
    S = np.diag([1.0 / s, 1.0 / s, 1.0])
    Hm = np.linalg.inv(S) @ warp @ S                           # back to full-size pixels
    return _nearest_rotation(KI @ Hm @ K), "ecc", float(cc)


def _rotvec(R):
    return cv2.Rodrigues(R)[0].ravel()


def _rot(v):
    return cv2.Rodrigues(np.asarray(v, float).reshape(3, 1))[0]


def bridge(pose_a, pose_b, frames, imgs):
    """pose_a / pose_b: dict(R=3x3 world-to-camera, center=3). frames: the frame numbers between a and b (exclusive),
    imgs: dict frame -> image for a, the frames between, and b (keys a..b). Returns {frame: dict(R, center, how)}."""
    a, b = min(imgs), max(imgs)
    Ra, Rb = np.array(pose_a["R"]), np.array(pose_b["R"])
    Ca, Cb = np.array(pose_a["center"]), np.array(pose_b["center"])
    chain, R, hows = [], Ra, []
    for f in range(a, b):
        T, how, _ = turn_between(imgs[f], imgs[f + 1])
        if T is None:
            T, how = np.eye(3), "hold"
        R = T @ R                                              # camera f+1 = turn applied after camera f
        chain.append(R)
        hows.append(how)
    E = Rb @ chain[-1].T                                       # what the chain is missing at b
    ev = _rotvec(E)
    n = b - a
    out = {}
    for k, f in enumerate(range(a + 1, b), start=1):
        Rk = _rot(ev * (k / n)) @ chain[k - 1]
        t = k / n
        s = t * t * (3 - 2 * t)                                # smoothstep: starts and ends at rest
        out[f] = dict(R=Rk.tolist(), center=(Ca + (Cb - Ca) * s).tolist(), how=hows[k - 1])
    return out, float(np.degrees(np.linalg.norm(ev)))


def angle_between(R1, R2):
    """Angle in degrees between two rotations."""
    return float(np.degrees(np.linalg.norm(_rotvec(np.array(R1) @ np.array(R2).T))))

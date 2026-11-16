"""Cross-check the solved opening shot: where is the boy in 3D, and how far does the camera go round HIM.

The boy = 3D points whose observations in a frame fall on his orange hair. Also: how much the camera itself
turns (yaw about the scene's up) between frames, which does not depend on choosing an orbit centre.
"""
import json
import sys

import cv2
import numpy as np
import pycolmap

D = sys.argv[1]
J = json.load(open(f"{D}/path.json"))
import os
rec = max((pycolmap.Reconstruction(f"{D}/sparse/{m}") for m in os.listdir(f"{D}/sparse")), key=lambda r: r.num_reg_images())
up = np.array(J["up"])
byname = {im.name: im for im in rec.images.values()}
pts = []
for f in (24, 30, 36, 42, 48):
    im = byname[f"f{f:04d}.png"]
    hsv = cv2.cvtColor(cv2.imread(f"{D}/images/f{f:04d}.png"), cv2.COLOR_BGR2HSV)
    hair = ((hsv[..., 0] >= 9) & (hsv[..., 0] <= 13) & (hsv[..., 1] > 140) & (hsv[..., 2] > 120))   # our curls (the rope is hue 2-4)
    for p2 in im.points2D:
        if p2.has_point3D():
            x, y = int(round(p2.xy[0])), int(round(p2.xy[1]))
            if 0 <= y < hair.shape[0] and 0 <= x < hair.shape[1] and hair[y, x]:
                pts.append(rec.points3D[p2.point3D_id].xyz)
pts = np.array(pts)
print("3D points on the boy's hair:", len(pts))
boy = np.median(pts, axis=0)
flat = lambda v: v - up * (v @ up)
P = {int(k): v for k, v in J["path"].items()}
C = lambda f: np.array(P[f]["center"])
yawdir = lambda f: flat(np.array(P[f]["R"]).T @ np.array([0, 0, 1.0]))
ang = lambda a, b: np.degrees(np.arctan2(np.cross(a, b) @ up, a @ b))
for a, b in ((0, 54), (9, 54), (20, 54), (9, 30), (30, 54)):
    print(f"frames {a:2d}-{b:2d}: round the boy {ang(flat(C(a) - boy), flat(C(b) - boy)):+6.1f} deg | camera turns {ang(yawdir(a), yawdir(b)):+6.1f} deg"
          f" | distance to boy x{np.linalg.norm(C(b) - boy) / np.linalg.norm(C(a) - boy):.2f}"
          f" | rise {((C(b) - C(a)) @ up) / np.linalg.norm(C(a) - boy):+.2f} of distance")
print("per 3 frames, degrees round the boy:", [round(float(ang(flat(C(f) - boy), flat(C(f + 3) - boy))), 1) for f in range(0, 52, 3)])

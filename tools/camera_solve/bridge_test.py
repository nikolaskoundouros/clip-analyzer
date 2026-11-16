"""Test of the whip bridge against COLMAP itself (15 Nov 2026).

Take frames that COLMAP DID solve, pretend a block of them is a whip gap (drop their poses), bridge the block with
tools/camera_path.bridge, and measure the angle between the bridged rotation and COLMAP's own rotation for every
dropped frame. Also reports the chain's end error E (how far the OpenCV chain alone drifted before being pinned).
Usage: python bridge_test.py MODELS.json FRAMES_DIR   (FRAMES_DIR: f0000.png ... at 540x960)
Prints one line per gap size and block; the numbers are the evidence, nothing here is tuned to pass."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import camera_path as cp

models = json.load(open(sys.argv[1]))["models"]
FR = sys.argv[2]
rows = []
for m in models:
    P = {int(f): v for f, v in m["path"].items()}
    fs = sorted(P)
    for gap in (5, 10, 18):                                   # 5 = the short gaps, 18 = the long whip (83-100)
        for start in range(fs[0], fs[-1] - gap - 1, max(gap, 12)):
            a, b = start, start + gap + 1
            if a not in P or b not in P or any(f not in P for f in range(a, b + 1)):
                continue
            imgs = {f: cv2.imread(os.path.join(FR, f"f{f:04d}.png")) for f in range(a, b + 1)}
            out, drift = cp.bridge(P[a], P[b], list(range(a + 1, b)), imgs)
            err = [cp.angle_between(out[f]["R"], P[f]["R"]) for f in range(a + 1, b)]
            cen = [float(np.linalg.norm(np.array(out[f]["center"]) - np.array(P[f]["center"]))) for f in range(a + 1, b)]
            span = float(np.linalg.norm(np.array(P[b]["center"]) - np.array(P[a]["center"])))
            hows = [out[f]["how"] for f in range(a + 1, b)]
            rows.append((gap, a, b, drift, max(err), float(np.mean(err)), max(cen), span, hows.count("sift"), hows.count("ecc"), hows.count("hold")))
            print(f"gap {gap:2d} frames {a}-{b}: chain drift before pinning {drift:5.1f} deg | bridged-vs-COLMAP rotation error mean {np.mean(err):5.2f} max {max(err):5.2f} deg"
                  f" | centre error max {max(cen):.3f} (solved centres {span:.3f} apart) | turns by sift/ecc/hold {hows.count('sift')}/{hows.count('ecc')}/{hows.count('hold')}")
for gap in (5, 10, 18):
    r = [x for x in rows if x[0] == gap]
    if r:
        print(f"SUMMARY gap {gap}: {len(r)} blocks, mean error {np.mean([x[5] for x in r]):.2f} deg, worst {max(x[4] for x in r):.2f} deg")

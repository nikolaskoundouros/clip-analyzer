"""Camera motion (2D and 3D), cuts and lighting on EVERY frame of a clip.

Per frame, against the frame before it (on a 360 x 640 grey copy, burned-in caption band masked out):
  2D   zoom, pan, tilt, roll from tracked points (Lucas-Kanade + RANSAC similarity fit);
  whip when tracking fails in blur, the move is read from the whole picture (phase correlation);
       a consistent shift = WHIP with direction and speed, none = CUT;
  3D   against the frame 3 frames earlier: the camera's real move, from the essential matrix:
       travel direction (right / down / forward), turn (yaw, pitch) and PARALLAX, which is how much
       near and far points move differently. Parallax only appears when the camera itself travels,
       so it tells a dolly from a lens zoom and an orbit from a pan;
  label one of: hold, pan/tilt, zoom in/out (lens), dolly in/out, truck left/right, crane up/down,
       orbit left/right, whip <dir>, cut; and, where the 3D read is not trusted (a moving subject
       fills the frame, or the points do not fit one rigid move), only what the picture does:
       push in, pull out, move left/right/up/down;
  light brightness, contrast, warmth (red / blue), saturation; sharpness;
  flow the mean point motion in each cell of a 6 x 10 grid (for drawing arrows over the video).
Units: zoom % size change per second (+ = in); pan_x % of width per second (+ = camera right);
pan_y % of height per second (+ = camera down); yaw / pitch degrees per second (+ = right / down);
parallax % of width per second; travel = unit direction in the camera's own axes.
Usage: python3 scan.py CLIP OUT_DIR [--band TOP,BOTTOM,LEFT,RIGHT | --no-mask] [--focal F]
       band = caption band as fractions of the frame; F = focal length in frame widths (default 1.3).
Writes OUT_DIR/scan.csv and OUT_DIR/flow.json, and progress lines "PROGRESS done total" on stdout.
"""
import collections
import csv
import json
import math
import sys

import cv2
import numpy as np

W, H = 360, 640
PW, PH = 90, 160
GX, GY = 6, 10
MIN_INL = 25
BASE = 3                 # frames between the pair used for the 3D read
PARALLAX_MIN = 3.0       # % of width per second: below this the camera is not travelling
ZOOM_MIN, PAN_MIN, YAW_ORBIT = 6.0, 5.0, 5.0
FLAT_MIN = 0.85          # this share of points fitting one flat move = no travel
RIGID_MIN = 0.20         # at least this share must fit one rigid 3D move to trust it
FOV_DEG = 42.0           # horizontal field of view assumed for the default focal length
TURN_MAX = 60.0          # degrees per second; a faster turn while tracking is a bad read


def label_of(r):
    """One word for what the camera is doing on this frame."""
    if r["status"] == "whip":
        ax = "right" if r["pan_x_per_s"] > 0 else "left"
        if abs(r["pan_y_per_s"]) > abs(r["pan_x_per_s"]):
            ax = "down" if r["pan_y_per_s"] > 0 else "up"
        return "whip " + ax
    if r["status"] != "track":
        return r["status"]
    if r["mode3d"] == "unsure":            # the 3D read is not trusted here: say only what the picture does
        if abs(r["zoom_per_s"]) >= ZOOM_MIN and abs(r["zoom_per_s"]) >= max(abs(r["pan_x_per_s"]), abs(r["pan_y_per_s"])):
            return "push in" if r["zoom_per_s"] > 0 else "pull out"
        if abs(r["pan_x_per_s"]) >= PAN_MIN or abs(r["pan_y_per_s"]) >= PAN_MIN:
            if abs(r["pan_x_per_s"]) >= abs(r["pan_y_per_s"]):
                return "move right" if r["pan_x_per_s"] > 0 else "move left"
            return "move down" if r["pan_y_per_s"] > 0 else "move up"
        return "hold"
    if r["mode3d"] == "travel" and r["parallax"] >= PARALLAX_MIN:
        tx, ty, tz = r["travel_x"], r["travel_y"], r["travel_z"]
        if abs(tz) >= abs(tx) and abs(tz) >= abs(ty):
            return "dolly in" if tz > 0 else "dolly out"
        if abs(tx) >= abs(ty):
            side = "right" if tx > 0 else "left"
            # an orbit turns against its travel so the subject stays put; in a truck the whole picture slides.
            # The turn must also be at least as big as the slide seen in the picture (in degrees).
            slide_deg = abs(r["pan_x_per_s"]) / 100 * FOV_DEG
            turning_against = r["yaw_per_s"] * tx < 0 and abs(r["yaw_per_s"]) >= max(YAW_ORBIT, slide_deg)
            return ("orbit " if turning_against else "truck ") + side
        return "crane down" if ty > 0 else "crane up"
    if abs(r["zoom_per_s"]) >= ZOOM_MIN and abs(r["zoom_per_s"]) >= abs(r["pan_x_per_s"]):
        return "zoom in" if r["zoom_per_s"] > 0 else "zoom out"
    if abs(r["pan_x_per_s"]) >= PAN_MIN or abs(r["pan_y_per_s"]) >= PAN_MIN:
        if abs(r["pan_x_per_s"]) >= abs(r["pan_y_per_s"]):
            return "pan right" if r["pan_x_per_s"] > 0 else "pan left"
        return "tilt down" if r["pan_y_per_s"] > 0 else "tilt up"
    if abs(r["zoom_per_s"]) >= ZOOM_MIN:
        return "zoom in" if r["zoom_per_s"] > 0 else "zoom out"
    return "hold"


def main():
    clip, out = sys.argv[1], sys.argv[2]
    band = (0.715, 0.785, 0.08, 0.92)
    if "--band" in sys.argv:
        band = tuple(float(v) for v in sys.argv[sys.argv.index("--band") + 1].split(","))
    if "--no-mask" in sys.argv:
        band = None
    focal = float(sys.argv[sys.argv.index("--focal") + 1]) if "--focal" in sys.argv else 1.3
    K = np.array([[focal * W, 0, W / 2], [0, focal * W, H / 2], [0, 0, 1]], dtype=np.float64)

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    keep = np.full((H, W), 255, np.uint8)
    if band:
        keep[int(band[0] * H):int(band[1] * H), int(band[2] * W):int(band[3] * W)] = 0
    ptop = int(band[0] * PH) if band else PH
    win = cv2.createHanningWindow((PW, ptop), cv2.CV_32F)

    def off_band(p):
        if not band:
            return np.ones(len(p), bool)
        yy, xx = p[:, 1] / H, p[:, 0] / W
        return ~((yy > band[0]) & (yy < band[1]) & (xx > band[2]) & (xx < band[3]))

    def tracks(g0, g1):
        p0 = cv2.goodFeaturesToTrack(g0, maxCorners=500, qualityLevel=0.01, minDistance=7, mask=keep)
        if p0 is None or len(p0) < 12:
            return None, None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=4)
        ok = st.ravel() == 1
        a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
        ok2 = off_band(b)
        return a[ok2], b[ok2]

    history = collections.deque(maxlen=BASE)
    prev = prev_small = prev_hist = None
    rows, flows, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        small = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], keep, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        ps = cv2.GaussianBlur(cv2.resize(g, (PW, PH), interpolation=cv2.INTER_AREA), (5, 5), 0)[:ptop].astype(np.float32)
        m = keep > 0
        lum = g[m].astype(np.float32) / 255
        row = dict(t=round(i / fps, 4), status="track", label="hold", zoom_per_s=0.0, pan_x_per_s=0.0, pan_y_per_s=0.0,
                   roll_deg_per_s=0.0, travel_x=0.0, travel_y=0.0, travel_z=0.0, yaw_per_s=0.0, pitch_per_s=0.0,
                   parallax=0.0, mode3d="unsure", flat=0.0, rigid=0.0, subject_per_s=0.0, inliers=0,
                   sharp=round(float(cv2.Laplacian(g, cv2.CV_32F)[m].var()), 1), luma=round(float(lum.mean()), 4),
                   contrast=round(float(np.percentile(lum, 95) - np.percentile(lum, 5)), 4),
                   warmth=round(float(small[:, :, 2][m].mean() / max(small[:, :, 0][m].mean(), 1e-3)), 4),
                   sat=round(float(hsv[:, :, 1][m].mean()) / 255, 4))
        cells = []
        if prev is not None:
            hcorr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            a, b = tracks(prev, g)
            model, n = None, 0
            if a is not None and len(a) >= 12:
                model, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5,
                                                          maxIters=2000, confidence=0.99)
                if model is not None:
                    inl = inl.ravel().astype(bool)
                    n = int(inl.sum())
            if model is not None and n >= MIN_INL and hcorr > 0.55:
                sa, sb = model[0, 0], model[1, 0]
                cx, cy = W / 2, H / 2
                mx = model[0, 0] * cx + model[0, 1] * cy + model[0, 2] - cx
                my = model[1, 0] * cx + model[1, 1] * cy + model[1, 2] - cy
                row.update(zoom_per_s=round((math.hypot(sa, sb) - 1.0) * fps * 100, 3),
                           pan_x_per_s=round(-mx / W * fps * 100, 3), pan_y_per_s=round(-my / H * fps * 100, 3),
                           roll_deg_per_s=round(math.degrees(math.atan2(sb, sa)) * fps, 3), inliers=n)
                rest = ~inl
                if rest.sum() >= 5:
                    pred = a[rest] @ model[:, :2].T + model[:, 2]
                    row["subject_per_s"] = round(float(np.median(np.linalg.norm(b[rest] - pred, axis=1))) / W * fps * 100, 3)
                d = b - a                                   # arrows: mean point motion per grid cell
                ci = np.clip((a[:, 0] / W * GX).astype(int), 0, GX - 1) + GX * np.clip((a[:, 1] / H * GY).astype(int), 0, GY - 1)
                for c in np.unique(ci):
                    sel = ci == c
                    if sel.sum() >= 2:
                        cells.append([int(c), int(round(float(np.median(d[sel, 0])) / W * 2000)),
                                      int(round(float(np.median(d[sel, 1])) / W * 2000))])
                # ---- 3D read over a wider gap
                if len(history) == BASE:
                    a3, b3 = tracks(history[0], g)
                    if a3 is not None and len(a3) >= 30:
                        E, em = cv2.findEssentialMat(a3, b3, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
                        if E is not None and E.shape == (3, 3):
                            _, R, tv, pm = cv2.recoverPose(E, a3, b3, K, mask=em)
                            ein = pm.ravel() > 0
                            Hm, hm = cv2.findHomography(a3, b3, cv2.RANSAC, 1.5)
                            flat = float(hm.sum()) / len(a3) if Hm is not None else 0.0     # share one flat picture move explains
                            rigid = float((em.ravel() > 0).sum()) / len(a3)                    # share one rigid 3D move explains
                            row.update(flat=round(flat, 3), rigid=round(rigid, 3))
                            if flat >= FLAT_MIN:
                                row["mode3d"] = "flat"          # no depth effect: the camera turns or zooms, it does not travel
                            elif rigid >= RIGID_MIN and ein.sum() >= 20:
                                C = (-R.T @ tv).ravel()
                                Rc = R.T
                                yaw = math.degrees(math.atan2(Rc[0, 2], Rc[2, 2])) * fps / BASE
                                pitch = math.degrees(math.asin(max(-1.0, min(1.0, Rc[1, 2])))) * fps / BASE
                                if abs(yaw) <= TURN_MAX and abs(pitch) <= TURN_MAX:
                                    pr = cv2.perspectiveTransform(a3[ein].reshape(-1, 1, 2).astype(np.float64), Hm).reshape(-1, 2)
                                    res = np.linalg.norm(b3[ein] - pr, axis=1)
                                    row.update(mode3d="travel", parallax=round(float(np.percentile(res, 85)) / W * 100 * fps / BASE, 3),
                                               travel_x=round(float(C[0]), 3), travel_y=round(float(C[1]), 3), travel_z=round(float(C[2]), 3),
                                               yaw_per_s=round(yaw, 3), pitch_per_s=round(pitch, 3))
            else:
                (dx, dy), resp = cv2.phaseCorrelate(prev_small, ps, win)
                row["inliers"] = n
                if resp > 0.12 and math.hypot(dx, dy) > 1.0 and hcorr > 0.35:
                    row.update(status="whip", pan_x_per_s=round(-dx / PW * fps * 100, 3), pan_y_per_s=round(-dy / PH * fps * 100, 3))
                elif hcorr < 0.80:
                    row["status"] = "cut"
                else:
                    row["status"] = "lost"
            if row["status"] != "track":
                history.clear()
        row["label"] = label_of(row)
        rows.append(row)
        flows.append(cells)
        history.append(g)
        prev, prev_small, prev_hist = g, ps, hist
        i += 1
        if i % 60 == 0:
            print(f"PROGRESS {i} {total}", flush=True)
    with open(f"{out}/scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump({"gx": GX, "gy": GY, "unit": 2000, "frames": flows}, open(f"{out}/flow.json", "w"), separators=(",", ":"))
    c = collections.Counter(r["status"] for r in rows)
    print(f"PROGRESS {i} {i}")
    print(f"frames {len(rows)} fps {fps:.0f}  " + "  ".join(f"{k} {v} ({100 * v / len(rows):.0f}%)" for k, v in c.most_common()))


if __name__ == "__main__":
    main()

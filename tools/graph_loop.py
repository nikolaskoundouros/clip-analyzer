"""Closed loop that makes OUR motion graph match the reference's (15 Nov 2026, Nick: "try the identical once").

Each round: render our clip on the pod (camera driven by the reference's analyzer readings + corr.json),
read OUR render back through the same analyzer, and add the difference (reference - ours) to the commands:
    corr[k+1] = corr[k] + GAIN * (reference_reading - our_reading)       per frame, for zoom, pan x, pan y
Round 0 is the render we already have (match A: SMOOTH=0 WHIPRAW=1, no correction), so the first new render is round 1.
Prints, per round: correlation of the zoom and pan graphs with the reference, the 1/3 s picture-change correlation
(tools/camera_solve/../smooth_score logic), pan direction flips per second, and how many frames the analyzer
reads as whip / cut. Writes clips/run021-loop-<k>/ (so the page shows each round) and loop_log.json.
Usage: python3 graph_loop.py ROUNDS [GAIN]
"""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

CA = "/Users/nick/Projects/clip-analyzer"
RUN = "/Users/nick/Projects/blender-funnel/runs/021-reference-shots"
REPO = "/Users/nick/run021loop-repo/runs/021-reference-shots"
ZOOMS = f"{RUN}/review/zooms"
REF_VIDEO = f"{ZOOMS}/1_reference_0-12.6s.mp4"
KEYS = ("zoom_per_s", "pan_x_per_s", "pan_y_per_s")
N, FPS = 379, 30.0
ROUNDS = int(sys.argv[1])
GAIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
sh = lambda *a, **k: subprocess.run(list(a), check=True, **k)


def load(clip):
    return json.load(open(f"{CA}/clips/{clip}/data.json"))


def series(D, key):
    return np.array([(D["series"][key][i] or 0.0) for i in range(N)], float)


def change(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-t", "12.62", "-i", path, "-vf", "scale=90:160,format=gray", "-f", "rawvideo", "-"],
                       capture_output=True).stdout
    f = np.frombuffer(r, np.uint8).reshape(-1, 160, 90).astype(float)
    return np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))


def blocks(v, k=10):
    n = len(v) // k * k
    return v[:n].reshape(-1, k).mean(axis=1)


REF = load("dino")
ref_series = {k: series(REF, k) for k in KEYS}
ref_change = change(REF_VIDEO)


def score(clip, video):
    D = load(clip)
    o = {k: series(D, k) for k in KEYS}
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    px = o["pan_x_per_s"]
    v = [x for x, st in zip(px, D["status"]) if st == "track" and abs(x) > 8]
    flips = sum(1 for x, y in zip(v, v[1:]) if x * y < 0)
    ch = change(video)
    n = min(len(ch), len(ref_change))
    return dict(zoom_corr=round(c(ref_series["zoom_per_s"], o["zoom_per_s"]), 2),
                pan_corr=round(c(ref_series["pan_x_per_s"], px), 2),
                tilt_corr=round(c(ref_series["pan_y_per_s"], o["pan_y_per_s"]), 2),
                change_corr_third_s=round(c(blocks(ch[:n]), blocks(ref_change[:n])), 2),
                pan_flips_per_s=round(flips / 12.62, 1),
                whip_frames=D["status"].count("whip"), cut_frames=D["status"].count("cut"), series=o, status=D["status"])


def analyze(clip, video):
    d = f"{CA}/clips/{clip}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(video, f"{d}/clip.mp4")
    sh("python3", "-I", f"{CA}/scan.py", f"{d}/clip.mp4", d, "--band", "0.715,0.785,0.08,0.92", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sh("python3", "-I", f"{CA}/build_data.py", d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def update(corr, ours_series, ours_status):
    new = {}
    for k in KEYS:
        d = ref_series[k] - ours_series[k]
        ok = np.array([(s != "cut" and r != "cut") for s, r in zip(ours_status, REF["status"][:N])])
        d = np.where(ok, d, 0.0)
        new[k] = list(np.clip(np.array(corr[k]) + GAIN * d, -600, 600))
    return new


log = []
r0 = score("run021-match-a", f"{ZOOMS}/match_A.mp4")                    # round 0 = the render we already have
print("ROUND 0 (match A, no correction)", {k: v for k, v in r0.items() if k not in ("series", "status")}, flush=True)
log.append(dict(round=0, **{k: v for k, v in r0.items() if k not in ("series", "status")}))
corr = update({k: [0.0] * N for k in KEYS}, r0["series"], r0["status"])
for rnd in range(1, ROUNDS + 1):
    json.dump(corr, open(f"{RUN}/corr.json", "w"))
    sh("sh", "/Users/nick/funnel_loop_iter.sh")
    frames = f"{REPO}/out_loop/frames"
    if not os.path.isdir(frames) or len(os.listdir(frames)) < N:
        print(f"ROUND {rnd}: render failed, see /Users/nick/funnel_loop_iter.log", flush=True)
        break
    video = f"{ZOOMS}/loop_{rnd}.mp4"
    sh("ffmpeg", "-v", "error", "-y", "-framerate", "30", "-start_number", "1", "-i", f"{frames}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", video)
    analyze(f"run021-loop-{rnd}", video)
    r = score(f"run021-loop-{rnd}", video)
    line = {k: v for k, v in r.items() if k not in ("series", "status")}
    print(f"ROUND {rnd}", line, flush=True)
    log.append(dict(round=rnd, **line))
    json.dump(log, open(f"{ZOOMS}/loop_log.json", "w"), indent=1)
    corr = update(corr, r["series"], r["status"])
print("LOOP DONE", flush=True)

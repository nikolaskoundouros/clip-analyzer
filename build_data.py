"""Turns scan.csv + flow.json into data.json, the one file the page loads.

Adds: a smoothed label per frame, the list of moves (runs of one label), the transitions with whip
direction and speed, and a summary. Usage: python3 build_data.py CLIP_DIR
"""
import collections
import csv
import json
import sys

import numpy as np

d = sys.argv[1]
R = list(csv.DictReader(open(f"{d}/scan.csv")))
n = len(R)
num = lambda k: np.array([float(r[k]) for r in R])
t = num("t")
fps = round(1.0 / (t[1] - t[0]), 3)
raw = [r["label"] for r in R]
status = [r["status"] for r in R]
label = list(raw)
for i in range(n):                      # majority vote over 7 frames, never across a transition
    if status[i] != "track":
        continue
    a, b = i, i
    while a > 0 and i - a < 3 and status[a - 1] == "track":
        a -= 1
    while b < n - 1 and b - i < 3 and status[b + 1] == "track":
        b += 1
    label[i] = collections.Counter(raw[a:b + 1]).most_common(1)[0][0]
zoom, px, py = num("zoom_per_s"), num("pan_x_per_s"), num("pan_y_per_s")
moves, i = [], 0
while i < n:
    j = i
    while j + 1 < n and label[j + 1] == label[i]:
        j += 1
    seg = slice(i, j + 1)
    kind = label[i].split()[0]
    mv = dict(a=i, b=j, start=round(float(t[i]), 2), dur=round((j - i + 1) / fps, 2), label=label[i],
              zoom_total=round(float((np.prod(1 + zoom[seg] / 100 / fps) - 1) * 100), 1),
              zoom_peak=round(float(np.abs(zoom[seg]).max()), 0),
              pan=round(float(px[seg].sum() / fps), 1), tilt=round(float(py[seg].sum() / fps), 1),
              speed=round(float(max(np.abs(px[seg]).max(), np.abs(py[seg]).max())), 0),
              parallax=round(float(np.median(num("parallax")[seg])), 1))
    moves.append(mv)
    i = j + 1
big = [m for m in moves if m["dur"] >= 0.2 or m["label"].startswith(("whip", "cut"))]
count = collections.Counter(m["label"] for m in big)
secs = collections.Counter()
for l in label:
    secs[l] += 1 / fps
cols = ["zoom_per_s", "pan_x_per_s", "pan_y_per_s", "roll_deg_per_s", "travel_x", "travel_y", "travel_z", "yaw_per_s",
        "pitch_per_s", "parallax", "luma", "warmth", "contrast", "sat", "sharp"]
flow = json.load(open(f"{d}/flow.json"))
out = dict(fps=fps, n=n, duration=round(n / fps, 2), label=label, status=status, mode3d=[r["mode3d"] for r in R],
           series={c: [round(float(v), 2) for v in num(c)] for c in cols}, moves=big,
           summary=dict(read=round(100 * sum(s in ("track", "whip") for s in status) / n, 1),
                        count=dict(count.most_common()), seconds={k: round(v, 1) for k, v in secs.most_common()}),
           flow=flow)
json.dump(out, open(f"{d}/data.json", "w"), separators=(",", ":"))
print(f"data.json: {n} frames, {len(big)} moves; " + ", ".join(f"{k} {v:.0f}s" for k, v in secs.most_common(8)))

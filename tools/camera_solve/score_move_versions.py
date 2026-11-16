"""Score the move-test versions against the reference's "over" to "rope" (frames 82-124), the same way for both:
each 43-frame stretch becomes its own clip and goes through the clip analyzer (scan.py + build_data.py).
Per version: sideways speed curve match (correlation), average gap, whip frames found, zoom curve match.
Also a sheet with every frame: reference on top, then each version. Usage: python3 score_mt.py REPO_RUN_DIR OUT_DIR"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

RUN, OUT = sys.argv[1], sys.argv[2]
REF = "/Users/nick/Desktop/My Mom Cracked The Dinosaur 🦖.mp4"
CA = "/Users/nick/Projects/clip-analyzer"
run = lambda *a: subprocess.run(list(a), check=True, capture_output=True)
os.makedirs(OUT, exist_ok=True)
clips = {}
d = f"{CA}/clips/mt-reference"
os.makedirs(d, exist_ok=True)
run("ffmpeg", "-y", "-i", REF, "-vf", "select=between(n\\,82\\,124),scale=360:640,setsar=1", "-vsync", "0", "-an", "-c:v", "libx264",
    "-pix_fmt", "yuv420p", "-crf", "16", "-r", "30", f"{d}/clip.mp4")
clips["reference"] = d
for v in ("A", "B", "C"):
    fr = f"{RUN}/out_mt_{v}/frames"
    if not os.path.isdir(fr) or len(os.listdir(fr)) < 43:
        print(v, "missing frames")
        continue
    d = f"{CA}/clips/mt-{v}"
    os.makedirs(d, exist_ok=True)
    run("ffmpeg", "-y", "-framerate", "30", "-start_number", "83", "-i", f"{fr}/f%04d.png", "-frames:v", "43", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "16", f"{d}/clip.mp4")
    run("ffmpeg", "-y", "-i", f"{d}/clip.mp4", "-vf", "setpts=4*PTS", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        f"{OUT}/ours_{v}_quarter_speed.mp4")
    run("cp", f"{d}/clip.mp4", f"{OUT}/ours_{v}.mp4")
    clips[v] = d
for name, d in clips.items():
    run("python3", "-I", f"{CA}/scan.py", f"{d}/clip.mp4", d, "--band", "0.715,0.785,0.08,0.92")
    run("python3", "-I", f"{CA}/build_data.py", d)
data = {k: json.load(open(f"{d}/data.json")) for k, d in clips.items()}
R = data["reference"]
n = min(x["n"] for x in data.values())
ser = lambda D, k: np.array([D["series"][k][i] or 0 for i in range(n)])
print("frames compared:", n)
print(f"reference: whip frames {sum(1 for s in R['status'][:n] if s == 'whip')}, cut frames {sum(1 for s in R['status'][:n] if s == 'cut')}")
best = None
for k, D in data.items():
    if k == "reference":
        continue
    px, rx = ser(D, "pan_x_per_s"), ser(R, "pan_x_per_s")
    pz, rz = ser(D, "zoom_per_s"), ser(R, "zoom_per_s")
    sm = lambda x: np.convolve(x, np.ones(3) / 3, "same")
    c_pan = float(np.corrcoef(sm(px), sm(rx))[0, 1])
    c_zoom = float(np.corrcoef(sm(pz), sm(rz))[0, 1])
    gap = float(np.mean(np.abs(px - rx)))
    whips = sum(1 for s in D["status"][:n] if s == "whip")
    both = sum(1 for i in range(n) if D["status"][i] == "whip" and R["status"][i] == "whip")
    score = c_pan + 0.5 * c_zoom - gap / 400
    print(f"version {k}: sideways curve match {c_pan:+.2f}, zoom curve match {c_zoom:+.2f}, average sideways gap {gap:.0f} %/s, "
          f"whip frames {whips} ({both} on the same frames as the reference), score {score:+.2f}")
    if best is None or score > best[1]:
        best = (k, score)
print("closest version:", best[0] if best else None)
# sheet: every frame, reference on top
W, H = 90, 160
rows = list(clips)
sheet = Image.new("RGB", (43 * (W + 2) + 40, len(rows) * (H + 4) + 16), (15, 15, 15))
dr = ImageDraw.Draw(sheet)
for r, k in enumerate(rows):
    tmp = f"{OUT}/_f_{k}"
    os.makedirs(tmp, exist_ok=True)
    run("ffmpeg", "-y", "-i", f"{clips[k]}/clip.mp4", "-vf", f"scale={W}:{H}", "-start_number", "0", f"{tmp}/%03d.png")
    dr.text((2, 16 + r * (H + 4) + H // 2), k[:3], fill=(255, 255, 0))
    for i in range(43):
        p = f"{tmp}/{i:03d}.png"
        if os.path.exists(p):
            sheet.paste(Image.open(p), (40 + i * (W + 2), 16 + r * (H + 4)))
        if r == 0 and i % 3 == 0:
            dr.text((40 + i * (W + 2), 2), f"{(82 + i) / 30:.2f}s", fill=(200, 200, 200))
sheet.save(f"{OUT}/every_frame_ref_and_versions.jpg", quality=85)
print("sheet saved")

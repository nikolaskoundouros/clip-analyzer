"""0 to 10 s check: two separate videos (reference, ours), a frame sheet every 1/3 s for checking by eye,
and the clip analyzer on ours vs the reference and v2 over frames 0-302.
Usage: python3 check_10s.py FRAMES_DIR OUT_DIR
"""
import collections
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

FR, OUT = sys.argv[1], sys.argv[2]
REF = "/Users/nick/Desktop/My Mom Cracked The Dinosaur 🦖.mp4"
CA = "/Users/nick/Projects/clip-analyzer"
N = 303
run = lambda *a: subprocess.run(list(a), check=True, capture_output=True)
os.makedirs(OUT, exist_ok=True)
run("ffmpeg", "-y", "-t", f"{N / 30:.3f}", "-i", REF, "-vf", "fps=30,scale=360:640,setsar=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "-crf", "18", "-c:a", "aac", f"{OUT}/1_reference_0-10s.mp4")
run("ffmpeg", "-y", "-framerate", "30", "-start_number", "1", "-i", f"{FR}/f%04d.png", "-t", f"{N / 30:.3f}", "-i", REF,
    "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-shortest",
    f"{OUT}/2_ours_0-10s.mp4")

# sheet: every 10 frames, reference on top, ours under
tmp = os.path.join(OUT, "_r")
os.makedirs(tmp, exist_ok=True)
run("ffmpeg", "-y", "-i", REF, "-vf", f"select=lt(n\\,{N}),scale=120:213", "-vsync", "0", "-start_number", "0", f"{tmp}/r_%04d.png")
fr = list(range(0, N, 10))
W, H = 120, 213
sheet = Image.new("RGB", (16 * (W + 3), 2 * ((len(fr) + 15) // 16) * (H + 4) + 40), (15, 15, 15))
d = ImageDraw.Draw(sheet)
for k, f in enumerate(fr):
    x, row = (k % 16) * (W + 3), k // 16
    y = row * 2 * (H + 4) + 14 * (row + 1)
    d.text((x + 2, y - 13), f"{f / 30:.2f}s", fill=(255, 255, 0))
    sheet.paste(Image.open(f"{tmp}/r_{f:04d}.png").convert("RGB"), (x, y))
    sheet.paste(Image.open(f"{FR}/f{f + 1:04d}.png").convert("RGB").resize((W, H)), (x, y + H + 2))
sheet.save(f"{OUT}/sheet_0-10s.jpg", quality=85)

clip = os.path.join(CA, "clips", "run021-10s")
os.makedirs(clip, exist_ok=True)
run("cp", f"{OUT}/2_ours_0-10s.mp4", f"{clip}/clip.mp4")
run("python3", "-I", f"{CA}/scan.py", f"{clip}/clip.mp4", clip, "--band", "0.715,0.785,0.08,0.92")
run("python3", "-I", f"{CA}/build_data.py", clip)
R = json.load(open(f"{CA}/clips/dino/data.json"))
O = json.load(open(f"{clip}/data.json"))
V2 = json.load(open(f"{CA}/clips/run021-ours-v2/data.json"))
n = min(N, O["n"])
fam = lambda l: l.split()[0]
print(f"frames compared: {n} (0 to {n / 30:.2f} s)")
for name, D in (("reference", R), ("v2 driven", V2), ("solved", O)):
    tr = [i for i in range(n) if D["mode3d"][i] == "travel"]
    side = np.mean([np.hypot(D["series"]["travel_x"][i], D["series"]["travel_y"][i]) > abs(D["series"]["travel_z"][i]) for i in tr])
    still = sum(1 for i in range(n) if D["status"][i] == "track" and abs(D["series"]["zoom_per_s"][i] or 0) < 5
                and abs(D["series"]["pan_x_per_s"][i] or 0) < 2 and abs(D["series"]["pan_y_per_s"][i] or 0) < 2)
    line = f"{name:10s} sideways/up travel {side * 100:3.0f}% of {len(tr)} trusted frames | near-still frames {still}"
    if D is not R:
        for fam_name in ("orbit", "crane", "dolly"):
            c = collections.Counter(fam(D["label"][i]) for i in range(n) if fam(R["label"][i]) == fam_name)
            line += f" | ref {fam_name} frames read as {fam_name}: {c[fam_name]}/{sum(c.values())}"
        zr = [R["series"]["zoom_per_s"][i] or 0 for i in range(n)]
        zo = [D["series"]["zoom_per_s"][i] or 0 for i in range(n)]
        sm = lambda x: np.convolve(x, np.ones(5) / 5, "same")
        line += f" | zoom curve match (5-frame) {np.corrcoef(sm(zr), sm(zo))[0, 1]:.2f}"
    print(line)

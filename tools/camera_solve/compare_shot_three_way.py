"""Step 3 check, opening shot (frames 0-54): reference vs v2 (driven) vs solved camera.

Makes: s1_three_way.mp4 (reference | v2 | solved), fbf_s1_solved.jpg (every 3rd frame, three rows),
and prints the boy's size in frame (orange hair) per frame for all three, with the correlation to the
reference. Also prepares DIR/ours_images and masks for running the same COLMAP solve on our render.
Usage: python3 check_s1.py SOLVED_FRAMES_DIR
"""
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

FR = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
R21 = "/private/tmp/claude-501/-Users-nick/a3dc1fd1-1e29-443f-acf0-49dc9d39846e/scratchpad/run021"
REF = "/Users/nick/Desktop/My Mom Cracked The Dinosaur 🦖.mp4"
run = lambda *a: subprocess.run(list(a), check=True)

ours = os.path.join(HERE, "ours_s1_solved.mp4")
run("ffmpeg", "-v", "error", "-y", "-framerate", "30", "-start_number", "1", "-i", os.path.join(FR, "f%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", ours)
run("ffmpeg", "-v", "error", "-y", "-t", "1.834", "-i", REF, "-t", "1.834", "-i", os.path.join(R21, "ours_v2.mp4"), "-i", ours,
    "-filter_complex", "[0:v]fps=30,scale=360:640,setsar=1[a];[1:v]fps=30,scale=360:640,setsar=1[b];[2:v]fps=30,scale=360:640,setsar=1[c];[a][b][c]hstack=inputs=3[v]",
    "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-shortest",
    os.path.join(HERE, "s1_three_way.mp4"))

os.makedirs(os.path.join(HERE, "s", "o"), exist_ok=True)
run("ffmpeg", "-v", "error", "-y", "-i", ours, "-vf", "scale=135:240", "-start_number", "1", os.path.join(HERE, "s", "o", "s_%04d.png"))
W, H = 135, 240
rows = [("REFERENCE", lambda f: f"{R21}/fbf/r_{f + 1:04d}.png"), ("v2 driven", lambda f: f"{R21}/fbf/o_{f + 1:04d}.png"),
        ("SOLVED", lambda f: f"{HERE}/s/o/s_{f + 1:04d}.png")]
fr = list(range(0, 55, 3))
sheet = Image.new("RGB", (len(fr) * (W + 3), 3 * (H + 16) + 18), (15, 15, 15))
d = ImageDraw.Draw(sheet)
for k, f in enumerate(fr):
    d.text((k * (W + 3) + 3, 3), f"{f} ({f / 30:.2f}s)", fill=(255, 255, 0))
    for r, (label, path) in enumerate(rows):
        sheet.paste(Image.open(path(f)).convert("RGB").resize((W, H)), (k * (W + 3), 18 + r * (H + 16)))
        if k == 0:
            d.text((3, 18 + r * (H + 16) + H + 2), label, fill=(255, 255, 255))
sheet.save(os.path.join(HERE, "fbf_s1_solved.jpg"), quality=88)


def hair(path):
    im = cv2.imread(path)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] <= 18) | (hsv[..., 0] >= 172)) & (hsv[..., 1] > 150) & (hsv[..., 2] > 110)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(np.sqrt(m.sum()) / im.shape[0])


hs = {label: np.array([hair(path(f)) for f in range(55)]) for label, path in rows}
for label, v in hs.items():
    print(f"{label:10s} hair size  0.0s {v[0]:.3f}  0.3s {v[9]:.3f}  1.0s {v[30]:.3f}  1.6s {v[48]:.3f}  1.8s {v[54]:.3f}"
          f"   correlation with reference {np.corrcoef(hs['REFERENCE'], v)[0, 1]:.2f}   mean ratio to reference {np.mean(v / np.maximum(hs['REFERENCE'], 1e-3)):.2f}")

# images + masks for solving our own render with COLMAP, same settings as the reference
oi, om = os.path.join(HERE, "ours_images"), os.path.join(HERE, "ours_masks")
os.makedirs(oi, exist_ok=True)
os.makedirs(om, exist_ok=True)
run("ffmpeg", "-v", "error", "-y", "-i", ours, "-vf", "scale=540:960", "-start_number", "0", os.path.join(oi, "f%04d.png"))
for f in sorted(os.listdir(oi)):
    cv2.imwrite(os.path.join(om, f + ".png"), np.full((960, 540), 255, np.uint8))
print("ours_images:", len(os.listdir(oi)))

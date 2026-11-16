"""Burned-in captions: when each one appears, how long it stays, and what it says.

On every frame the caption letters are picked out in the caption band (white pixels right next to a
dark outline). A new caption starts when the letter shapes change. One picture per caption is read
with Apple's on-device text reader (ocr.js), so the words and their capitals come from the
picture itself, not from the speech.
Writes CLIP_DIR/captions.json. Usage: python3 captions.py CLIP_DIR [--band TOP,BOTTOM,LEFT,RIGHT]
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

d = Path(sys.argv[1])
band = (0.70, 0.80, 0.05, 0.95)
if "--band" in sys.argv:
    band = tuple(float(v) for v in sys.argv[sys.argv.index("--band") + 1].split(","))
video = next(p for p in d.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".webm"))
cap = cv2.VideoCapture(str(video))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
MIN_PIX, SAME = 60, 0.45          # letters must cover this many pixels; shapes this alike = the same caption
kernel = np.ones((5, 5), np.uint8)
segs, cur, prev_mask, i = [], None, None, 0
crops = {}
while True:
    ok, fr = cap.read()
    if not ok:
        break
    h, w = fr.shape[:2]
    crop = fr[int(band[0] * h):int(band[1] * h), int(band[2] * w):int(band[3] * w)]
    small = cv2.resize(crop, (486, int(486 * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask = (g > 225) & (cv2.erode(g, kernel) < 70)
    area = int(mask.sum())
    if area < MIN_PIX:
        if cur:
            segs.append(cur)
        cur, prev_mask = None, None
    else:
        same = False
        if cur is not None and prev_mask is not None:
            inter = int((mask & prev_mask).sum())
            same = inter / max(1, min(area, int(prev_mask.sum()))) >= SAME
        if same:
            cur["b"] = i
            if area > cur["area"]:                      # keep the clearest picture of this caption
                cur["area"] = area
                crops[cur["a"]] = crop.copy()
        else:
            if cur:
                segs.append(cur)
            cur = dict(a=i, b=i, area=area)
            crops[i] = crop.copy()
        prev_mask = mask
    i += 1
if cur:
    segs.append(cur)
segs = [s for s in segs if s["b"] - s["a"] + 1 >= 2]

with tempfile.TemporaryDirectory() as tmp:
    for s in segs:
        c = crops[s["a"]]
        cv2.imwrite(f"{tmp}/{s['a']:06d}.png", cv2.resize(c, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
    r = subprocess.run(["osascript", "-l", "JavaScript", str(Path(__file__).resolve().parent / "ocr.js"), tmp], capture_output=True, text=True)
    read = {}
    for line in r.stdout.splitlines():
        name, _, text = line.partition("\t")
        read[int(name.split(".")[0])] = text.strip()
    if r.returncode != 0:
        print("text reader failed:", r.stderr[-200:])

out = []
for s in segs:
    text = read.get(s["a"], "")
    if out and text and out[-1]["text"] == text and s["a"] - out[-1]["b"] <= 2:      # the same words, briefly interrupted
        out[-1]["b"] = s["b"]
        continue
    out.append(dict(a=s["a"], b=s["b"], text=text))
for c in out:
    c["start"] = round(c["a"] / fps, 3)
    c["dur"] = round((c["b"] - c["a"] + 1) / fps, 3)
read_ok = [c for c in out if c["text"]]
words = [len(c["text"].split()) for c in read_ok]
durs = [c["dur"] for c in out]
caps = [c for c in read_ok if c["text"].isupper() and any(ch.isalpha() for ch in c["text"])]
stats = dict(count=len(out), read=len(read_ok), median_dur=round(float(np.median(durs)), 2) if durs else 0,
             mean_words=round(float(np.mean(words)), 2) if words else 0, max_words=max(words) if words else 0,
             all_caps=len(caps), on_screen=round(100 * sum(c["b"] - c["a"] + 1 for c in out) / max(i, 1), 1))
json.dump(dict(fps=fps, band=band, captions=out, stats=stats), open(d / "captions.json", "w"), separators=(",", ":"))
print("captions.json:", stats)
print("first:", " | ".join(c["text"] for c in out[:14]))
print("all caps:", [c["text"] for c in caps][:10])

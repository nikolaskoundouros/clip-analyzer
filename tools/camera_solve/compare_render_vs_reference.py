"""Run 021 check: our render vs the reference, by eye (side-by-side video) and by number (clip analyzer).

1. frames -> ours.mp4 at the reference's frame rate, side_by_side.mp4 (reference 0-12.62 s left, ours right, reference sound)
2. clip analyzer (scan.py + build_data.py, same caption band as the reference) on ours -> clips/run021-ours
3. for every reference move: the zoom and pan it adds up to in the reference vs in ours over the same frames
Usage: python3 compare.py FRAMES_DIR
"""
import json
import math
import os
import shutil
import subprocess
import sys

FR = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
REF = "/Users/nick/Desktop/My Mom Cracked The Dinosaur 🦖.mp4"
CA = "/Users/nick/Projects/clip-analyzer"
OUT = os.path.join(CA, "clips", "run021-ours")
FPS = 30.0
run = lambda *a: subprocess.run(list(a), check=True)

ours = os.path.join(HERE, "ours.mp4")
run("ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-start_number", "1", "-i", os.path.join(FR, "f%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", ours)
run("ffmpeg", "-v", "error", "-y", "-t", "12.62", "-i", REF, "-i", ours, "-filter_complex",
    "[0:v]scale=360:640,setsar=1[a];[1:v]scale=360:640,setsar=1[b];[a][b]hstack=inputs=2[v]",
    "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-shortest",
    os.path.join(HERE, "side_by_side.mp4"))

os.makedirs(OUT, exist_ok=True)
shutil.copy(ours, os.path.join(OUT, "clip.mp4"))
run("python3", "-I", f"{CA}/scan.py", os.path.join(OUT, "clip.mp4"), OUT, "--band", "0.715,0.785,0.08,0.92")
run("python3", "-I", f"{CA}/build_data.py", OUT)

R = json.load(open(f"{CA}/clips/dino/data.json"))
O = json.load(open(f"{OUT}/data.json"))
n = min(O["n"], 379)


def totals(D, a, b):
    s = D["series"]
    z = 1.0
    px = py = 0.0
    for i in range(a, min(b, n - 1) + 1):
        if D["status"][i] in ("cut", "lost"):
            continue
        if D["status"][i] == "track":
            z *= 1 + (s["zoom_per_s"][i] or 0) / 100 / FPS
        px += (s["pan_x_per_s"][i] or 0) / FPS
        py += (s["pan_y_per_s"][i] or 0) / FPS
    return round((z - 1) * 100, 1), round(px, 1), round(py, 1)


rows = []
for m in R["moves"]:
    if m["a"] >= n or m["label"] == "cut":
        continue
    rz, rx, ry = totals(R, m["a"], m["b"])
    oz, ox, oy = totals(O, m["a"], m["b"])
    olab = max(set(O["label"][m["a"]:m["b"] + 1]), key=O["label"][m["a"]:m["b"] + 1].count)
    rows.append(dict(t=m["start"], dur=m["dur"], ref_label=m["label"], our_label=olab,
                     ref_zoom=rz, our_zoom=oz, ref_pan=rx, our_pan=ox, ref_tilt=ry, our_tilt=oy))
big = [r for r in rows if abs(r["ref_zoom"]) >= 20]
same_dir = sum(1 for r in big if r["ref_zoom"] * r["our_zoom"] > 0)
whips_r = sum(1 for r in rows if r["ref_label"].startswith("whip"))
whips_o = sum(1 for r in rows if r["ref_label"].startswith("whip") and r["our_label"].startswith("whip"))
corr = lambda a, b: (sum((x - sum(a) / len(a)) * (y - sum(b) / len(b)) for x, y in zip(a, b)) /
                     math.sqrt(sum((x - sum(a) / len(a)) ** 2 for x in a) * sum((y - sum(b) / len(b)) ** 2 for y in b) + 1e-12))
zr = [R["series"]["zoom_per_s"][i] or 0 for i in range(n)]
zo = [O["series"]["zoom_per_s"][i] or 0 for i in range(n)]
pr = [R["series"]["pan_x_per_s"][i] or 0 for i in range(n)]
po = [O["series"]["pan_x_per_s"][i] or 0 for i in range(n)]
summary = dict(frames=n, big_zoom_moves=len(big), same_direction=same_dir, ref_whips=whips_r, whips_read_as_whips_in_ours=whips_o,
               zoom_curve_correlation=round(corr(zr, zo), 2), pan_curve_correlation=round(corr(pr, po), 2))
json.dump(dict(summary=summary, moves=rows), open(os.path.join(HERE, "compare.json"), "w"), indent=1)
print("SUMMARY", json.dumps(summary))
for r in big:
    print(f"{r['t']:5.2f}s {r['ref_label']:<12} ref zoom {r['ref_zoom']:+7.1f}%  ours {r['our_zoom']:+7.1f}%  ({r['our_label']})")

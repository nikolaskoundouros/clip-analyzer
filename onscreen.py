"""Who is on screen, and how big, on every frame of a clip.

Faces: OpenCV's YuNet detector (MIT, models/face_detection_yunet_2023mar.onnx) on every frame; each face
is turned into a 128-number fingerprint by OpenCV's SFace (Apache 2.0, models/face_recognition_sface_2021dec.onnx).
Fingerprints are grouped into characters across the whole clip (average-link clustering, cut at cosine
similarity 0.28), so "character 1" is the same person every time they appear. Characters are numbered by
how much screen time they have; naming them (boy, Mom, guard) is a separate step.
Per frame: the main character (largest face), every face's box, and the shot size from the main face's
height in the frame: ECU >= 0.40, CU 0.20-0.40, MED 0.09-0.20, WIDE below 0.09 or no readable face
(fitted on one clip, the dinosaur reference: confirm on a second clip before trusting).
Writes CLIP_DIR/onscreen.json. Usage: python3 onscreen.py CLIP_DIR [--every N]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
d = Path(sys.argv[1])
every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1
video = next(p for p in d.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".webm") and p.stem == "clip")
cap = cv2.VideoCapture(str(video))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = 540
det = None
rec = cv2.FaceRecognizerSF.create(str(ROOT / "models" / "face_recognition_sface_2021dec.onnx"), "")
SIZES = ((0.40, "ECU"), (0.20, "CU"), (0.09, "MED"))   # halfway between the dinosaur clip's by-eye medians (ECU 0.46, CU 0.31, MED 0.13, WIDE 0.07)

frames, feats = [], []           # feats: (frame index, face index, fingerprint)
i = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if i % every:
        i += 1
        continue
    h0, w0 = fr.shape[:2]
    small = cv2.resize(fr, (W, round(W * h0 / w0)))
    if det is None:
        det = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "", (small.shape[1], small.shape[0]),
                                        score_threshold=0.6, nms_threshold=0.3, top_k=20)
    _, faces = det.detect(small)
    row = dict(f=i, faces=[])
    for k, fc in enumerate(faces if faces is not None else []):
        x, y, w, h = (float(v) for v in fc[:4])
        crop = rec.alignCrop(small, fc)
        feats.append((len(frames), k, rec.feature(crop).flatten()))
        row["faces"].append(dict(box=[round(x / small.shape[1], 4), round(y / small.shape[0], 4), round(w / small.shape[1], 4),
                                      round(h / small.shape[0], 4)], score=round(float(fc[14]), 3)))
    frames.append(row)
    if i % 300 == 0:
        print(f"PROGRESS {i}", flush=True)
    i += 1

# group fingerprints into characters: average-link clustering (a group is judged by ALL its members, so
# it cannot drift and chain everyone together, which the first, running-average version did on the dinosaur
# clip: one group held 1,830 of 1,889 face frames). Cut at similarity 0.28, measured on that clip's answer key:
# same character 0.35-0.56, different characters 0.14-0.23 on average.
from scipy.cluster.hierarchy import fcluster, linkage
CUT = 0.28
V = np.array([v / np.linalg.norm(v) for _, _, v in feats])
labels = fcluster(linkage(V, method="average", metric="cosine"), t=1 - CUT, criterion="distance") if len(V) > 1 else np.ones(len(V), int)
groups = {}
for n, g in enumerate(labels):
    groups.setdefault(int(g), []).append(n)
members = sorted(groups.values(), key=len, reverse=True)
who = {}
for rank, ms in enumerate(members):
    for m in ms:
        who[(feats[m][0], feats[m][1])] = rank + 1          # character 1 = most screen time

for fi, row in enumerate(frames):
    for k, face in enumerate(row["faces"]):
        face["character"] = who[(fi, k)]
    big = max(row["faces"], key=lambda q: q["box"][3], default=None)
    row["main"] = big["character"] if big else None
    hh = big["box"][3] if big else 0.0
    row["size"] = next((name for lim, name in SIZES if hh >= lim), "WIDE")
counts = {}
for row in frames:
    if row["main"]:
        counts[row["main"]] = counts.get(row["main"], 0) + 1
np.save(d / "onscreen_faces.npy", np.array([np.r_[fr2, k2, v2 / np.linalg.norm(v2)] for fr2, k2, v2 in feats]))   # row: frame-row, face, fingerprint
json.dump(dict(fps=fps, every=every, thresholds=dict(same_character=CUT, sizes=SIZES), characters=len(members),
               frames_per_character=counts, frames=frames), open(d / "onscreen.json", "w"), separators=(",", ":"))
print(f"onscreen.json: {len(frames)} frames, {len(feats)} faces, {len(members)} characters; main-character frames: "
      f"{dict(sorted(counts.items(), key=lambda kv: -kv[1])[:8])}; no face: {sum(1 for r in frames if not r['main'])}")

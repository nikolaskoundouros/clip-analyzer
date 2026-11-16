"""Names the sounds in a clip with the AudioSet tagger (MIT/ast-finetuned-audioset, 527 sound classes).

Runs on CLIP_DIR/stems/no_vocals.m4a (the voice split off) when it exists, else on the clip's own
audio. Slides a 1 s window in 0.25 s steps and keeps, per window, the best-scoring classes.
Music and speech classes are kept but marked, so the page can leave them out of the effects.
Writes CLIP_DIR/labels.json. Run with the demucs venv python (it has torch and transformers):
  ~/.venvs/demucs/bin/python label_sound.py CLIP_DIR
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
WIN, STEP, SR = 1.0, 0.5, 16000
WATCH = ("Whoosh, swoosh, swish", "Crack", "Breaking", "Smash, crash", "Thump, thud", "Slap, smack", "Wood", "Crushing",
         "Shatter", "Whip", "Rumble", "Bang", "Slam", "Sound effect", "Wind", "Rustle", "Whack, thwack", "Explosion", "Scream", "Footsteps")
d = Path(sys.argv[1])
stem = d / "stems" / "no_vocals.m4a"
src = stem if stem.exists() else next(p for p in d.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".webm"))
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                     capture_output=True, check=True).stdout
x = np.frombuffer(raw, dtype=np.float32)
fe = ASTFeatureExtractor.from_pretrained(MODEL)
model = ASTForAudioClassification.from_pretrained(MODEL).eval()
names = [model.config.id2label[i] for i in range(model.config.num_labels)]
watch = [(n, names.index(n)) for n in WATCH if n in names]
print("watching", [n for n, _ in watch], flush=True)
BACKGROUND = ("music", "speech", "sing", "instrument", "song", "narration", "conversation", "tune", "soundtrack",
              "guitar", "piano", "drum", "bass", "orchestra", "synth", "keyboard", "choir", "vocal")
is_bg = np.array([any(k in n.lower() for k in BACKGROUND) for n in names])
starts = np.arange(0, max(len(x) / SR - WIN, 0) + 1e-6, STEP)
out = []
with torch.no_grad():
    for b in range(0, len(starts), 8):
        chunk = [x[int(s * SR):int((s + WIN) * SR)] for s in starts[b:b + 8]]
        inp = fe(chunk, sampling_rate=SR, return_tensors="pt")
        prob = torch.sigmoid(model(**inp).logits).numpy()
        for s, p in zip(starts[b:b + 8], prob):
            fx = np.where(is_bg, 0, p)
            top = np.argsort(fx)[::-1][:5]
            bg = np.argsort(np.where(is_bg, p, 0))[::-1][:2]
            out.append(dict(t=round(float(s), 2), fx=[[names[i], round(float(p[i]), 3)] for i in top if p[i] >= 0.03],
                            bg=[[names[i], round(float(p[i]), 3)] for i in bg],
                            watch=[round(float(p[i]), 3) for _, i in watch]))
        print(f"PROGRESS {min(b + 8, len(starts))} {len(starts)}", flush=True)
json.dump(dict(model=MODEL, win=WIN, step=STEP, watch=[n for n, _ in watch], source="effects only (voice split off)" if stem.exists() else "full mix",
               windows=out), open(d / "labels.json", "w"), separators=(",", ":"))
seen = {}
for w in out:
    for n, p in w["fx"][:2]:
        seen[n] = max(seen.get(n, 0), p)
print("labels.json:", len(out), "windows; strongest effect classes:", sorted(seen.items(), key=lambda kv: -kv[1])[:12])

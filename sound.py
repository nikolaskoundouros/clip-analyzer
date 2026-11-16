"""Sound effects, loudness and a spectrogram from a clip's audio, with no model: signal only.

Reads CLIP_DIR/clip.* (or a given audio file), works at 32 kHz mono with a 10 ms step, and finds:
  whoosh  a swell of hiss-like high sound (2 to 14 kHz) that rises and falls over 0.12 to 1.5 s;
          marked rising or falling by which way its pitch centre moves;
  crack   a very short, sharp, bright hit (dies away within about 80 ms);
  thud    a hit whose energy is mostly below 200 Hz;
  hit     any other sudden sound.
Speech is NOT removed here: with a voice-over, the s and sh sounds of the voice can be read as short
whooshes. If CLIP_DIR/stems/no_vocals.m4a exists (the voice split off), that file is analysed instead
and "source" in the output says so.
Writes CLIP_DIR/sound.json (per-video-frame loudness and band levels, the events) and
CLIP_DIR/spectrogram.png. Usage: python3 sound.py CLIP_DIR [FPS]
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from scipy import signal

d = Path(sys.argv[1])
fps = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
SR, HOP, NFFT = 32000, 320, 1024
stem = d / "stems" / "no_vocals.m4a"
src = stem if stem.exists() else next(p for p in d.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".webm"))
wav = d / "sound_32k.wav"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav)], check=True)
with wave.open(str(wav)) as w:
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
wav.unlink()
f, t, Z = signal.stft(x, fs=SR, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
P = np.abs(Z) ** 2
T = P.shape[1]
step = HOP / SR
db = lambda v: 10 * np.log10(np.maximum(v, 1e-12))
band = lambda lo, hi: P[(f >= lo) & (f < hi)].sum(axis=0)
low, mid, hi, air = band(20, 200), band(200, 2000), band(2000, 6000), band(6000, 14000)
total = low + mid + hi + air + 1e-12
bright = hi + air
smooth = lambda v, n: np.convolve(v, np.ones(n) / n, mode="same")

# ---- hits: sudden jumps in the spectrum (spectral flux on 40 log-spaced bands)
edges = np.geomspace(60, 14000, 41)
B = np.stack([P[(f >= a) & (f < b)].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
L = np.log1p(B * 1e4)
flux = np.maximum(L[:, 1:] - L[:, :-1], 0).sum(axis=0)
flux = np.concatenate([[0], flux])
med = signal.medfilt(flux, 101)
mad = signal.medfilt(np.abs(flux - med), 101) + 1e-6
z = (flux - med) / mad
peaks, _ = signal.find_peaks(z, height=6.0, distance=8)
events = []
for p in peaks:
    a, b = p, min(T - 1, p + 8)                     # the 80 ms after the hit
    peak_db = db(total[a:a + 3].max())
    tail_db = db(total[b:b + 3].mean()) if b + 3 <= T else peak_db
    share_low = float(low[a:a + 4].sum() / total[a:a + 4].sum())
    share_bright = float(bright[a:a + 4].sum() / total[a:a + 4].sum())
    kind = "hit"
    if share_low > 0.5:
        kind = "thud"
    elif share_bright > 0.30 and peak_db - tail_db > 10:
        kind = "crack"
    events.append(dict(t=round(p * step, 3), end=round((p + 8) * step, 3), kind=kind, strength=round(float(z[p]), 1),
                       level=round(float(peak_db), 1)))

# ---- whooshes: a rise and fall of bright, hiss-like sound
env = db(smooth(bright, 5))
floor = signal.medfilt(env, 151)                    # 1.5 s running middle
wp, props = signal.find_peaks(env - floor, prominence=8.0, width=(12, 150), rel_height=0.5)
cent = (f[:, None] * P).sum(axis=0) / (P.sum(axis=0) + 1e-12)
for p, li, ri, prom in zip(wp, props["left_ips"], props["right_ips"], props["prominences"]):
    a, b = int(li), int(ri)
    rise = float(np.median(cent[p:b + 1]) - np.median(cent[a:p + 1]))
    events.append(dict(t=round(a * step, 3), peak=round(p * step, 3), end=round(b * step, 3), kind="whoosh",
                       dir="rising" if rise > 150 else "falling" if rise < -150 else "level",
                       strength=round(float(prom), 1), level=round(float(env[p]), 1)))
events.sort(key=lambda e: e["t"])

# ---- per-video-frame levels for the lanes
n = int(np.ceil(len(x) / SR * fps))
idx = np.minimum((np.arange(T) * step * fps).astype(int), n - 1)
def per_frame(v):
    out = np.full(n, -90.0)
    s = np.zeros(n)
    c = np.zeros(n)
    np.add.at(s, idx, v)
    np.add.at(c, idx, 1)
    ok = c > 0
    out[ok] = db(s[ok] / c[ok])
    return [round(float(q), 1) for q in out]
series = dict(loud=per_frame(total), low=per_frame(low), mid=per_frame(mid), bright=per_frame(bright))

# ---- spectrogram picture (log frequency, 60 Hz to 14 kHz)
rows = 128
fe = np.geomspace(60, 14000, rows + 1)
S = np.stack([P[(f >= a) & (f < b)].mean(axis=0) if ((f >= a) & (f < b)).any() else np.zeros(T) for a, b in zip(fe[:-1], fe[1:])])
S = db(S)
S = np.clip((S - (S.max() - 75)) / 75, 0, 1)[::-1]
img = cv2.applyColorMap((S * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
img = cv2.resize(img, (min(T, 4000), rows * 2), interpolation=cv2.INTER_AREA)
cv2.imwrite(str(d / "spectrogram.png"), img)

kinds = {}
for e in events:
    kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
json.dump(dict(source="effects only (voice split off)" if stem.exists() else "full mix (voice included)",
               fps=fps, n=n, duration=round(len(x) / SR, 2), series=series, events=events, counts=kinds),
          open(d / "sound.json", "w"), separators=(",", ":"))
print(f"sound.json: {len(events)} events {kinds}; source: {'stem' if stem.exists() else 'full mix'}")

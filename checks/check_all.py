"""The checks behind verify.sh.
1. Motion: a test clip with ten known 3D camera moves must be labelled right on at least 75% of each move's frames.
2. Off switch: with CLIP_UPLOAD=0 the server refuses an upload (403) and still serves the page.
3. Upload: with uploads on, a posted clip is analysed to "ready", its data has every frame, the video
   can be seeked (a Range request answers 206), and a feedback verdict is saved.
4. Sound: the test clip gets a sound track with a noise swell at 2.0 s and a sharp click at 6.0 s; after
   upload the voice split runs, a whoosh is found near 2.0 s and a crack or hit near 6.0 s, and the
   effects-only track can be fetched. With CLIP_SOUND=0 those steps are skipped.
"""
import collections
import csv
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = [sys.executable, "-I"]
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        fails.append(name)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def serve(clip_dir, upload, sound=True):
    port = free_port()
    env = dict(os.environ, PORT=str(port), CLIP_DIR=str(clip_dir), CLIP_UPLOAD="1" if upload else "0", CLIP_SOUND="1" if sound else "0")
    p = subprocess.Popen(PY + [str(ROOT / "server.py")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/config", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    return p, f"http://127.0.0.1:{port}"


def post(url, body, headers=None):
    req = urllib.request.Request(url, data=body, method="POST", headers=headers or {})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    fx = ROOT / "checks" / "fixtures"
    subprocess.run(PY + [str(ROOT / "checks" / "synth3d.py"), str(fx / "far.jpg"), str(fx / "mid.jpg"), str(fx / "near.jpg"), str(tmp)],
                   check=True, stdout=subprocess.DEVNULL)
    (tmp / "scan").mkdir()
    subprocess.run(PY + [str(ROOT / "scan.py"), str(tmp / "moves3d.mp4"), str(tmp / "scan"), "--no-mask"], check=True, stdout=subprocess.DEVNULL)
    truth = list(csv.DictReader(open(tmp / "truth_moves3d.csv")))
    scan = list(csv.DictReader(open(tmp / "scan" / "scan.csv")))
    by = collections.defaultdict(list)
    for a, b in zip(truth, scan):
        by[a["label"]].append(b["label"])
    for name, got in by.items():
        got = got[4:]                      # the first frames of a move still see the one before it
        share = sum(g == name for g in got) / len(got)
        check(f"motion: {name}", share >= 0.75, f"{100 * share:.0f}% of frames right")

    # a sound track: quiet room tone, a noise swell peaking at 2.25 s (whoosh), a 15 ms bright click at 6.0 s (crack)
    sr, rng = 44100, np.random.default_rng(1)
    tt = np.arange(10 * sr) / sr
    snd = 0.003 * rng.standard_normal(len(tt))
    snd += 0.35 * rng.standard_normal(len(tt)) * np.exp(-((tt - 2.25) / 0.12) ** 2)
    click = (tt >= 6.0) & (tt < 6.015)
    snd[click] += 0.9 * np.sin(2 * np.pi * 3500 * tt[click]) * np.exp(-(tt[click] - 6.0) * 300)
    with wave.open(str(tmp / "test_sound.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(snd, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp / "moves3d.mp4"), "-i", str(tmp / "test_sound.wav"),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp / "moves3d_sound.mp4")], check=True)
    p, base = serve(tmp / "clips_nosound", upload=True, sound=False)
    try:
        code, body = post(base + "/api/upload?name=moves3d.mp4", (tmp / "moves3d_sound.mp4").read_bytes())
        cid = body.get("id")
        for _ in range(240):
            st = json.loads(urllib.request.urlopen(f"{base}/clips/{cid}/status.json").read() or b"{}")
            if st.get("state") in ("ready", "failed"):
                break
            time.sleep(0.5)
        check("sound off switch: clip still analysed", st.get("state") == "ready", str(st)[:100])
        check("sound off switch: no sound step ran", not (tmp / "clips_nosound" / cid / "sound.json").exists())
    finally:
        p.kill()

    p, base = serve(tmp / "clips_off", upload=False)
    try:
        code, body = post(base + "/api/upload?name=x.mp4", b"12345")
        check("off switch: upload refused", code == 403, f"HTTP {code}")
        check("off switch: page still served", urllib.request.urlopen(base + "/").status == 200)
        check("off switch: config says off", json.loads(urllib.request.urlopen(base + "/api/config").read())["upload"] is False)
    finally:
        p.kill()

    p, base = serve(tmp / "clips_on", upload=True)
    try:
        code, body = post(base + "/api/upload?name=moves3d.mp4", (tmp / "moves3d_sound.mp4").read_bytes())
        check("upload: accepted", code == 200 and "id" in body, f"HTTP {code}")
        cid, state = body.get("id"), {}
        for _ in range(600):
            try:
                state = json.loads(urllib.request.urlopen(f"{base}/clips/{cid}/status.json").read())
            except Exception:
                state = {}
            if state.get("state") in ("ready", "failed"):
                break
            time.sleep(0.5)
        check("upload: analysed to ready", state.get("state") == "ready", str(state)[:160])
        data = json.loads(urllib.request.urlopen(f"{base}/clips/{cid}/data.json").read())
        check("upload: data has every frame", data["n"] == 300 and len(data["label"]) == 300 and len(data["flow"]["frames"]) == 300, f"n={data['n']}")
        r = urllib.request.urlopen(urllib.request.Request(f"{base}/clips/{cid}/clip.mp4", headers={"Range": "bytes=100-199"}))
        check("upload: video can be seeked", r.status == 206 and len(r.read()) == 100, f"HTTP {r.status}")
        code, _ = post(base + "/api/feedback", json.dumps({"clip": cid, "start": 1.0, "label": "dolly in", "verdict": "wrong", "correct": "zoom in"}).encode())
        saved = urllib.request.urlopen(f"{base}/clips/{cid}/feedback.jsonl").read().decode()
        check("feedback: verdict saved", code == 200 and '"correct": "zoom in"' in saved)
        snd = json.loads(urllib.request.urlopen(f"{base}/clips/{cid}/sound.json").read())
        check("sound: voice split ran", snd["source"].startswith("effects only"), snd["source"])
        w = [e for e in snd["events"] if e["kind"] == "whoosh" and abs(e["peak"] - 2.25) <= 0.25]
        check("sound: whoosh found at 2.0 s", bool(w), f"{len(w)} near; all events {snd['counts']}")
        c = [e for e in snd["events"] if e["kind"] in ("crack", "hit") and abs(e["t"] - 6.0) <= 0.1]
        check("sound: crack found at 6.0 s", bool(c), ", ".join(f"{e['kind']}@{e['t']}" for e in snd["events"] if abs(e["t"] - 6) < 0.5))
        r = urllib.request.urlopen(urllib.request.Request(f"{base}/clips/{cid}/no_vocals.m4a", headers={"Range": "bytes=0-99"}))
        check("sound: effects-only track served", r.status == 206, f"HTTP {r.status}")
        check("captions: step ran", urllib.request.urlopen(f"{base}/clips/{cid}/captions.json").status == 200)
        code, _ = post(base + "/api/upload?name=notes.txt", b"hello")
        check("upload: wrong file type refused", code == 400, f"HTTP {code}")
    finally:
        p.kill()

print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)

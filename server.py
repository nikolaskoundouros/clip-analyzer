"""The clip-analyzer page: a local server on 127.0.0.1 (default port 8570). Nothing leaves this Mac.

  GET  /                      the page (web/index.html)
  GET  /api/config            {"upload": true|false}
  GET  /api/clips             the analysed clips and their state
  GET  /clips/<id>/<file>     clip.mp4 (with seeking), data.json, status.json, feedback.jsonl
  POST /api/upload?name=X     body = the video file; saves it and runs scan.py then build_data.py
  POST /api/feedback          {"clip","start","label","verdict","correct"} appended to the clip's feedback.jsonl
Off switches: CLIP_UPLOAD=0 refuses uploads (403); CLIP_SOUND=0 skips the voice split, sound and caption
steps for new uploads. Everything else keeps working.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
CLIPS = Path(os.environ.get("CLIP_DIR", ROOT / "clips"))
UPLOAD_ON = os.environ.get("CLIP_UPLOAD", "1") != "0"
SOUND_ON = os.environ.get("CLIP_SOUND", "1") != "0"           # off switch for the sound and caption steps
DEMUCS = os.environ.get("DEMUCS", str(Path.home() / ".venvs" / "demucs" / "bin" / "demucs"))
MAX_BYTES = 600 * 1024 * 1024
TYPES = {".m4a": "audio/mp4", ".png": "image/png", ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
         ".jsonl": "text/plain; charset=utf-8", ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}


def set_status(d, **kw):
    (d / "status.json").write_text(json.dumps(kw))


def has_audio(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def sound_steps(d, video):
    """Voice split, sound effects, captions. Each step that fails is noted; the clip stays usable."""
    notes = []
    if has_audio(video):
        try:
            set_status(d, state="splitting voice from effects", video=video.name)
            work = d / "stems" / "work"
            work.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "2", "-ar", "44100", str(work / "mix.wav")], check=True)
            subprocess.run([DEMUCS, "--two-stems", "vocals", "-n", "htdemucs", "-d", "cpu", "-o", str(work), str(work / "mix.wav")],
                           check=True, capture_output=True)
            for name in ("no_vocals", "vocals"):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(work / "htdemucs" / "mix" / f"{name}.wav"), "-c:a", "aac", "-b:a", "160k",
                                str(d / "stems" / f"{name}.m4a")], check=True)
            shutil.rmtree(work, ignore_errors=True)
        except Exception as e:
            notes.append(f"voice split failed ({str(e)[-120:]}); sound read from the full mix")
        try:
            set_status(d, state="finding sound effects", video=video.name)
            r = subprocess.run([sys.executable, "-I", str(ROOT / "sound.py"), str(d)], capture_output=True, text=True, check=True)
            notes.append(r.stdout.strip()[-160:])
        except Exception as e:
            notes.append(f"sound step failed: {str(e)[-160:]}")
    else:
        notes.append("no sound track")
    try:
        set_status(d, state="reading captions", video=video.name)
        r = subprocess.run([sys.executable, "-I", str(ROOT / "captions.py"), str(d)], capture_output=True, text=True, check=True)
        notes.append(r.stdout.strip().splitlines()[0][:160] if r.stdout.strip() else "captions done")
    except Exception as e:
        notes.append(f"captions step failed: {str(e)[-160:]}")
    return notes


def pipeline(d, video):
    try:
        set_status(d, state="scanning", done=0, total=0, video=video.name)
        p = subprocess.Popen([sys.executable, "-I", str(ROOT / "scan.py"), str(video), str(d)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = []
        for line in p.stdout:
            tail = (tail + [line.rstrip()])[-6:]
            m = re.match(r"PROGRESS (\d+) (\d+)", line)
            if m:
                set_status(d, state="scanning", done=int(m.group(1)), total=int(m.group(2)), video=video.name)
        if p.wait() != 0:
            raise RuntimeError("scan failed: " + " | ".join(tail))
        set_status(d, state="building", video=video.name)
        r = subprocess.run([sys.executable, "-I", str(ROOT / "build_data.py"), str(d)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("build failed: " + r.stderr[-300:])
        notes = [r.stdout.strip()[-160:]]
        if SOUND_ON:
            notes += sound_steps(d, video)
        set_status(d, state="ready", video=video.name, note=" | ".join(notes))
    except Exception as e:                                   # shown on the page, never swallowed
        set_status(d, state="failed", video=video.name, error=str(e)[-400:])


class Handler(BaseHTTPRequestHandler):
    server_version = "clip-analyzer"

    def log_message(self, fmt, *args):
        if "/status.json" not in (args[0] if args else ""):
            sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        if not path.is_file():
            if path.name == "feedback.jsonl" and path.parent.is_dir():      # no feedback yet is not an error
                self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
                return
            return self.send_json({"error": "not found"}, 404)
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
        self.send_response(code)
        self.send_header("Content-Type", TYPES.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self.send_file(ROOT / "web" / "index.html")
        if path == "/favicon.ico":
            self.send_response(204); self.end_headers()
            return
        if path == "/api/config":
            return self.send_json({"upload": UPLOAD_ON})
        if path == "/api/clips":
            out = []
            for d in sorted(CLIPS.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True):
                st = {"state": "ready"} if (d / "data.json").exists() else {"state": "unknown"}
                if (d / "status.json").exists():
                    st = json.loads((d / "status.json").read_text() or "{}")
                vids = [p.name for p in d.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".webm")]
                out.append({"id": d.name, "video": vids[0] if vids else None, "stems": (d / "stems" / "no_vocals.m4a").exists(), **st})
            return self.send_json(out)
        m = re.match(r"^/clips/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)$", path)
        if m and ".." not in path:
            f = CLIPS / m.group(1) / m.group(2)
            if m.group(2) in ("no_vocals.m4a", "vocals.m4a"):
                f = CLIPS / m.group(1) / "stems" / m.group(2)
            return self.send_file(f)
        m = re.match(r"^/web/([A-Za-z0-9._-]+)$", path)
        if m:
            return self.send_file(ROOT / "web" / m.group(1))
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        if u.path == "/api/upload":
            if not UPLOAD_ON:
                self.rfile.read(n) if n < 1 << 20 else None
                return self.send_json({"error": "uploads are switched off (CLIP_UPLOAD=0)"}, 403)
            name = parse_qs(u.query).get("name", ["clip.mp4"])[0]
            ext = Path(name).suffix.lower()
            if ext not in (".mp4", ".mov", ".webm"):
                return self.send_json({"error": "upload an .mp4, .mov or .webm file"}, 400)
            if n <= 0 or n > MAX_BYTES:
                return self.send_json({"error": "file is empty or over 600 MB"}, 400)
            slug = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")[:40] or "clip"
            d = CLIPS / slug
            if d.exists():
                d = CLIPS / f"{slug}-{time.strftime('%H%M%S')}"
            d.mkdir(parents=True)
            video = d / f"clip{ext}"
            with open(video, "wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(1 << 16, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            set_status(d, state="queued", video=video.name)
            threading.Thread(target=pipeline, args=(d, video), daemon=True).start()
            return self.send_json({"id": d.name})
        if u.path == "/api/feedback":
            try:
                fb = json.loads(self.rfile.read(n) or b"{}")
                d = CLIPS / re.sub(r"[^A-Za-z0-9._-]", "", str(fb.get("clip", "")))
                if not d.is_dir():
                    raise ValueError("unknown clip")
                rec = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "start": float(fb["start"]), "label": str(fb["label"])[:40],
                       "verdict": "right" if fb.get("verdict") == "right" else "wrong", "correct": str(fb.get("correct", ""))[:40]}
                with open(d / "feedback.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                return self.send_json({"saved": True})
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)
        self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8570))
    CLIPS.mkdir(parents=True, exist_ok=True)
    print(f"clip-analyzer on http://127.0.0.1:{port}  uploads {'on' if UPLOAD_ON else 'OFF'}  clips in {CLIPS}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

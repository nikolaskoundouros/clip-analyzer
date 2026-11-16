"""Words, subject and camera move, lined up: what is said, who is on screen, what the camera does.

Reads CLIP_DIR/captions.json (burned-in captions with times), transcript.vtt if present (sentence timings), onscreen.json (onscreen.py: main character and
shot size per frame), data.json (camera moves) and, if present, names.json ({"1": "mom", ...}: who each
detected character is; without it characters stay "character N").
Per caption: on-screen subject and size 0.25 s after the caption starts (the camera lands on what is said
within about half a second, read by eye), the camera move running then, and whether a cut or whip happens
from 0.3 s before the caption to 0.5 s after it starts. Quoted captions (a line someone says) are marked.
Writes CLIP_DIR/script.json and prints the rule checks. Usage: python3 script_align.py CLIP_DIR
"""
import json
import os
import random
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
C = json.load(open(d / "captions.json"))["captions"]
O = json.load(open(d / "onscreen.json"))
D = json.load(open(d / "data.json"))
names = json.load(open(d / "names.json")) if (d / "names.json").exists() else {}
fps = D["fps"]
LAG = float(os.environ.get("LAG", 0.25))   # seconds after a caption starts at which "on screen" is read
rows = {r["f"]: r for r in O["frames"]}
step = O.get("every", 1)


def onscreen(t):
    f = int(round(t * fps))
    for g in range(f, f - step - 1, -1):
        if g in rows:
            r = rows[g]
            return (names.get(str(r["main"]), f"character {r['main']}") if r["main"] else "none"), r["size"]
    return "none", "WIDE"


def move_at(f):
    m = next((m for m in D["moves"] if m["a"] <= f <= m["b"]), None)
    return (m["label"], m["zoom_total"], m["pan"]) if m else ("none", 0.0, 0.0)


def cut_near(t):
    a, b = int((t - 0.3) * fps), int((t + 0.5) * fps)
    return any(D["status"][f] in ("whip", "cut") for f in range(max(a, 0), min(b, D["n"] - 1) + 1))


# Sentence ends from the transcript (WebVTT, if CLIP_DIR/transcript.vtt exists): the text reader misses some
# closing quote marks on the burned-in captions, so a spoken line also ends where its sentence's cue ends.
cue_starts = []
if (d / "transcript.vtt").exists():
    for line in open(d / "transcript.vtt"):
        m = re.match(r"(\d+):(\d+\.\d+) -->", line.strip())
        if m:
            cue_starts.append(int(m.group(1)) * 60 + float(m.group(2)))
cue_of = lambda t: sum(1 for s0 in cue_starts if s0 <= t + 0.15)
out, inside, cue = [], False, -1
for c in C:
    if cue_of(c["start"]) != cue:
        cue, inside = cue_of(c["start"]), False
    text = c["text"]
    q = text.count('"') + text.count("\u201c") + text.count("\u201d")
    quoted = inside or q > 0
    if q % 2 == 1:                                   # captions use one plain mark to open and close a line
        inside = not inside
    t = c["start"] + LAG
    who, size = onscreen(t)
    mv = move_at(int(t * fps))
    out.append(dict(start=c["start"], end=round(c["start"] + c["dur"], 2), text=text, quoted=quoted, subject=who, size=size,
                    move=mv[0], move_zoom=mv[1], move_pan=mv[2], cut_near=cut_near(c["start"])))
json.dump(dict(names=names, rows=out), open(d / "script.json", "w"), indent=1)

# ---- rule checks, each against chance (the same measure at random caption times)
NAMED = {"mom": r"\b(mom|she|her|ma'am)\b", "boy": r"\b(i|me|my)\b", "guard": r"\b(guard|security|he)\b",
         "prop": r"\b(bone|leg|crack|dust)\b", "skeleton": r"\b(skeleton|t-rex|whole thing|pounds)\b"}
SIZE_WORDS = r"\b(entire|whole|crush|pounds|skeleton)\b"
FEEL_WORDS = r"(!|\bdread\b|\bscream|\bkid\b|\bstupid\b|\bmom\b\")"
random.seed(0)
def chance(test, n=2000):
    hits = 0
    for _ in range(n):
        t = random.uniform(0, D["n"] / fps - 0.5)
        hits += test(*onscreen(t + LAG))
    return hits / n
def report(name, sel, test):
    rs = [r for r in out if sel(r)]
    if rs:
        rate = sum(test(r["subject"], r["size"]) for r in rs) / len(rs)
        print(f"{name}: {rate * 100:.0f}% of {len(rs)} captions   (chance {chance(test) * 100:.0f}%)")
close = lambda s, z: z in ("ECU", "CU")
report("1. quoted lines are close-ups", lambda r: r["quoted"], close)
for who, pat in NAMED.items():
    test = (lambda s, z, who=who: s == who) if who in ("mom", "boy", "guard") else (lambda s, z: s == "none" or z == "WIDE")
    report(f"2. '{who}' words show {who if who in ('mom', 'boy', 'guard') else 'no face / wide'}",
           lambda r, pat=pat: not r["quoted"] and re.search(pat, r["text"].lower()), test)
report("3. size words are wide or faceless", lambda r: re.search(SIZE_WORDS, r["text"].lower()), lambda s, z: z == "WIDE" or s == "none")
report("4. feeling words are extreme close-ups", lambda r: re.search(FEEL_WORDS, r["text"].lower()), lambda s, z: z == "ECU")
cuts = sum(r["cut_near"] for r in out) / len(out)
print(f"captions with a cut or whip starting near them: {cuts * 100:.0f}% of {len(out)}")

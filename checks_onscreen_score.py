"""Score onscreen.json against a by-eye answer key (onscreen_truth.txt). Usage: python3 checks_onscreen_score.py CLIP_DIR"""
import collections, json, sys
d = sys.argv[1]
O = json.load(open(f"{d}/onscreen.json"))
rows = {r["f"]: r for r in O["frames"]}
truth = []
for line in open(f"{d}/onscreen_truth.txt"):
    if line[0] != "#" and line.strip():
        t, sub, size = line.split()
        truth.append((float(t), sub, size))
def near(f):
    for g in (f, f - 1, f + 1, f - 2, f + 2):
        if g in rows:
            return rows[g]
# naming: each character takes the single-person name it coincides with most often
votes = collections.defaultdict(collections.Counter)
for t, sub, size in truth:
    r = near(round(t * 30))
    if r and r["main"] and sub in ("boy", "mom", "guard"):
        votes[r["main"]][sub] += 1
name = {c: v.most_common(1)[0][0] for c, v in votes.items()}
print("character -> name (votes):", {c: (name[c], dict(v)) for c, v in sorted(votes.items())})
ok_sub = ok_size = near_size = 0
ORDER = ["ECU", "CU", "MED", "WIDE"]
miss = collections.Counter()
conf = collections.defaultdict(collections.Counter)
for t, sub, size in truth:
    r = near(round(t * 30))
    pred = name.get(r["main"], "other") if r["main"] else "none"
    good = (pred == sub) or (sub == "both" and pred in ("boy", "mom")) or (sub in ("prop", "skeleton") and (pred == "none" or r["size"] == "WIDE"))
    ok_sub += good
    conf[sub][pred] += 1
    if not good:
        miss[(sub, pred)] += 1
    if sub not in ("prop", "skeleton"):
        ok_size += r["size"] == size
        near_size += abs(ORDER.index(r["size"]) - ORDER.index(size)) <= 1
n = len(truth); n_people = sum(1 for _, s, _ in truth if s not in ("prop", "skeleton"))
print(f"subject right: {ok_sub}/{n} ({ok_sub / n * 100:.0f}%)")
print(f"shot size right: {ok_size}/{n_people} ({ok_size / n_people * 100:.0f}%), within one step: {near_size}/{n_people} ({near_size / n_people * 100:.0f}%)")
print("answer key -> detector:", {k: dict(v) for k, v in conf.items()})
print("most common misses (key, detector):", miss.most_common(6))

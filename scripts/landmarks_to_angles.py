"""Converts a MediaPipe landmark CSV (imagePoses_filtered.csv style) into joint-angle CSVs
matching exercise_angles.csv columns, one file per exercise.

Usage: python3 scripts/landmarks_to_angles.py <input.csv> "<exercise>" "<Label>" <out.csv>
Angles are 2D (x, y) like the in-app tracker; uses the more visible body side per frame.
"""
import ast, sys, math
import pandas as pd

src, ex, label, out = sys.argv[1:5]
SIDES = {"left": (11, 13, 15, 23, 25, 27, 31), "right": (12, 14, 16, 24, 26, 28, 32)}

def parse(s):
    d = {}
    for part in s.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            d[k.strip()] = float(v)
    return d

def ang(a, b, c):
    v1 = (a["x"] - b["x"], a["y"] - b["y"]); v2 = (c["x"] - b["x"], c["y"] - b["y"])
    m = math.hypot(*v1) * math.hypot(*v2)
    if m == 0: return None
    return math.degrees(math.acos(max(-1, min(1, (v1[0]*v2[0] + v1[1]*v2[1]) / m))))

df = pd.read_csv(src)
df = df[df["excercise"] == ex]
rows = []
for s in df["poseLandmarks"].dropna():
    try: lm = [parse(x) for x in ast.literal_eval(s)]
    except Exception: continue
    if len(lm) != 33: continue
    side = max(SIDES, key=lambda k: sum(lm[i].get("visibility", 0) for i in SIDES[k][:3]))
    sh, el, wr, hp, kn, an, ft = (lm[i] for i in SIDES[side])
    if min(sh.get("visibility", 0), el.get("visibility", 0), wr.get("visibility", 0)) < 0.3: continue
    vals = [ang(hp, sh, el), ang(sh, el, wr), ang(sh, hp, kn), ang(hp, kn, an), ang(kn, an, ft)]
    if any(v is None for v in vals): continue
    rows.append([side] + [round(v, 4) for v in vals] + [label])
pd.DataFrame(rows, columns=["Side", "Shoulder_Angle", "Elbow_Angle", "Hip_Angle", "Knee_Angle", "Ankle_Angle", "Label"]).to_csv(out, index=False)
print(label, len(rows), "frames ->", out)

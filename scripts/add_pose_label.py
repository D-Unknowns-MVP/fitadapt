"""Merges an extra exercise CSV (same columns as exercise_angles.csv) into src/data/pose-reference.json.

Usage: python3 scripts/add_pose_label.py src/data/situp_angles.csv "Sit Up" sit_ups Hip_Angle
Uses the same quantile / threshold / k-NN sampling rules as build_pose_reference.py.
"""
import json, sys
import pandas as pd

JOINTS = ["Shoulder_Angle", "Elbow_Angle", "Hip_Angle", "Knee_Angle", "Ankle_Angle"]
SAMPLES_PER_CLASS = 160
src, label, key, prim = sys.argv[1:5]
g = pd.read_csv(src)
g = g[g["Label"] == label]
path = "src/data/pose-reference.json"
out = json.load(open(path))
out["samples"] = [s for s in out["samples"] if s[5] != key]
if key not in out["labels"]:
    out["totalRows"] += int(len(g))
q = g[JOINTS].quantile([0.05, 0.15, 0.5, 0.85, 0.95])
ps = [0.05, 0.15, 0.5, 0.85, 0.95]
p5, p95 = float(q.loc[0.05, prim]), float(q.loc[0.95, prim])
span = p95 - p5
hist = pd.cut(g[prim], bins=range(0, 190, 10), include_lowest=True).value_counts(sort=False).tolist()
out["labels"][key] = {
    "label": label, "rows": int(len(g)), "primaryJoint": prim.replace("_Angle", "").lower(),
    "quantiles": {j.replace("_Angle", "").lower(): [round(float(q.loc[p, j]), 1) for p in ps] for j in JOINTS},
    "romRange": [round(p5, 1), round(p95, 1)],
    "thresholds": {"low": round(p5 + 0.3 * span, 1), "high": round(p95 - 0.3 * span, 1)},
    "histogram": [int(c) for c in hist],
}
for _, r in g.sample(n=min(SAMPLES_PER_CLASS, len(g)), random_state=42).iterrows():
    out["samples"].append([round(float(r[j]), 1) for j in JOINTS] + [key])
json.dump(out, open(path, "w"), separators=(",", ":"))
print("ok", key, out["labels"][key]["thresholds"], len(out["samples"]))

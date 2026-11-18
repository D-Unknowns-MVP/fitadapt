import reference from "@/data/pose-reference.json";
import type { PoseKey } from "./types";

/**
 * Pose analytics driven by the Exercise Detection Dataset (exercise_angles.csv).
 * - Joint angles are computed the same way as the dataset (shoulder, elbow, hip, knee, ankle).
 * - A k-NN classifier over dataset samples recognises which exercise is being performed.
 * - Rep thresholds and the 100% range-of-motion span come from per-exercise dataset quantiles.
 */

export type JointName = "shoulder" | "elbow" | "hip" | "knee" | "ankle";
export type SideName = "left" | "right";
export const JOINTS: JointName[] = ["shoulder", "elbow", "hip", "knee", "ankle"];

export interface LabelRef {
  label: string;
  rows: number;
  primaryJoint: JointName;
  quantiles: Record<JointName, number[]>;
  romRange: [number, number];
  thresholds: { low: number; high: number };
  histogram: number[];
}

export const POSE_REF = reference as unknown as {
  source: string;
  totalRows: number;
  labels: Record<PoseKey, LabelRef>;
  samples: [number, number, number, number, number, PoseKey][];
};

export interface Point {
  x: number;
  y: number;
  visibility?: number;
}

export function angle(a: Point, b: Point, c: Point): number {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const m = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y);
  if (m === 0) return 0;
  return (Math.acos(Math.min(1, Math.max(-1, dot / m))) * 180) / Math.PI;
}

const SIDES = {
  left: { shoulder: 11, elbow: 13, wrist: 15, hip: 23, knee: 25, ankle: 27, foot: 31 },
  right: { shoulder: 12, elbow: 14, wrist: 16, hip: 24, knee: 26, ankle: 28, foot: 32 },
};

/**
 * Returns joint angles for a selected body side.
 * If no side is specified, automatically selects the more visible side.
 */
export function jointAngles(
  lm: Point[],
  preferredSide?: SideName
): { angles: number[]; visibility: number; side: SideName } | null {
  if (!lm || lm.length < 33) return null;

  const vis = (s: typeof SIDES.left) =>
    Object.values(s).reduce(
      (a, i) => a + (lm[i]!.visibility ?? 1),
      0
    ) / Object.values(s).length;

  const leftVisibility = vis(SIDES.left);
  const rightVisibility = vis(SIDES.right);

  const sideName: SideName =
    preferredSide ??
    (leftVisibility >= rightVisibility ? "left" : "right");

  const side = SIDES[sideName];
  const p = (i: number) => lm[i]!;

  return {
    side: sideName,
    visibility: vis(side),
    angles: [
      angle(p(side.elbow), p(side.shoulder), p(side.hip)),
      angle(p(side.shoulder), p(side.elbow), p(side.wrist)),
      angle(p(side.shoulder), p(side.hip), p(side.knee)),
      angle(p(side.hip), p(side.knee), p(side.ankle)),
      angle(p(side.knee), p(side.ankle), p(side.foot)),
    ],
  };
}
export function classify(angles: number[], k = 7): { key: PoseKey; confidence: number } {
  const dists = POSE_REF.samples.map((s) => {
    let d = 0;
    for (let j = 0; j < 5; j++) d += ((s[j] as number) - (angles[j] ?? 0)) ** 2;
    return { d, key: s[5] };
  });
  dists.sort((a, b) => a.d - b.d);
  const votes = new Map<PoseKey, number>();
  for (const n of dists.slice(0, k)) votes.set(n.key, (votes.get(n.key) ?? 0) + 1);
  let best: PoseKey = dists[0]!.key;
  let bestV = 0;
  for (const [key, v] of votes) if (v > bestV) [best, bestV] = [key, v];
  return { key: best, confidence: bestV / k };
}

/** True when every joint sits inside the dataset's 5th–95th percentile band for that exercise. */
export function withinBand(key: PoseKey, angles: number[], tolerance = 10): boolean {
  const q = POSE_REF.labels[key].quantiles;
  return JOINTS.every((j, i) => {
    const a = angles[i] ?? 0;
    return a >= q[j][0]! - tolerance && a <= q[j][4]! + tolerance;
  });
}

export interface RepEvent {
  rom: number;
  form: number;
}

/**
 * Hysteresis rep counter on the exercise's primary joint.
 * A rep = leaving the start zone, crossing the opposite threshold, and returning.
 */
export class RepCounter {
  readonly ref: LabelRef;
  readonly jointIndex: number;
  private start: "low" | "high" | null = null;
  private crossed = false;
  private min = Infinity;
  private max = -Infinity;
  private frames = 0;
  private goodFrames = 0;
  reps = 0;
  lastRom = 0;

  constructor(readonly key: PoseKey) {
    this.ref = POSE_REF.labels[key];
    this.jointIndex = JOINTS.indexOf(this.ref.primaryJoint);
  }

  /** Feed one frame of joint angles; returns a RepEvent when a repetition completes. */
  push(angles: number[], formFrame?: boolean): RepEvent | null {
    const a = angles[this.jointIndex] ?? 0;
    const { low, high } = this.ref.thresholds;
    const zone = a <= low ? "low" : a >= high ? "high" : "mid";

    if (this.start === null) {
      if (zone !== "mid") this.start = zone;
      this.resetRep(a);
      return null;
    }
    this.min = Math.min(this.min, a);
    this.max = Math.max(this.max, a);
    this.frames++;
    if (formFrame ?? withinBand(this.key, angles)) this.goodFrames++;

    const opposite = this.start === "low" ? "high" : "low";
    if (zone === opposite) this.crossed = true;
    if (zone === this.start && this.crossed) {
      const [p5, p95] = this.ref.romRange;
      const rom = Math.max(0, Math.min(1, (this.max - this.min) / (p95 - p5)));
      const form = this.frames ? this.goodFrames / this.frames : 1;
      this.reps++;
      this.lastRom = rom;
      this.resetRep(a);
      return { rom, form };
    }
    return null;
  }

  private resetRep(a: number) {
    this.crossed = false;
    this.min = a;
    this.max = a;
    this.frames = 0;
    this.goodFrames = 0;
  }
}


// FITADAPT_DATASET_TRACKING
export type DatasetPoseKey =
  | "reverse_lunge"
  | "leg_press"
  | "burpee"
  | "kettlebell_swing"
  | "romanian_deadlift";

export interface DatasetPoseConfig {
  side: SideName;
  joint: "hip" | "knee";
  low: number;
  high: number;
  label: string;
}

export const DATASET_POSE_CONFIG: Record<DatasetPoseKey, DatasetPoseConfig> = {
  reverse_lunge: {
    side: "right", joint: "knee", low: 120, high: 165, label: "Reverse Lunge"
  },
  leg_press: {
    side: "left", joint: "knee", low: 100, high: 155, label: "Leg Press"
  },
  burpee: {
    side: "left", joint: "hip", low: 75, high: 155, label: "Burpee"
  },
  kettlebell_swing: {
    side: "left", joint: "hip", low: 115, high: 165, label: "Kettlebell Swing"
  },
  romanian_deadlift: {
    side: "right", joint: "hip", low: 90, high: 155, label: "Romanian Deadlift"
  },
};

export function getDatasetPoseConfig(key: PoseKey): DatasetPoseConfig | null {
  return key in DATASET_POSE_CONFIG
    ? DATASET_POSE_CONFIG[key as DatasetPoseKey]
    : null;
}

/** Require the three landmarks used by the selected joint angle to be visible. */
export function isDatasetJointVisible(
  lm: Point[],
  config: DatasetPoseConfig
): boolean {
  const indices =
    config.joint === "knee"
      ? config.side === "left" ? [23, 25, 27] : [24, 26, 28]
      : config.side === "left" ? [11, 23, 25] : [12, 24, 26];

  return indices.every((i) => (lm[i]?.visibility ?? 1) >= 0.65);
}

/**
 * Conservative angle-cycle counter for the five additional exercises.
 * These starting thresholds require live-camera calibration before research use.
 */
export class DatasetRepCounter {
  private start: "low" | "high" | null = null;
  private crossed = false;
  private min = Infinity;
  private max = -Infinity;
  private frames = 0;
  reps = 0;
  lastRom = 0;

  constructor(readonly key: DatasetPoseKey) {}

  push(angles: number[]): RepEvent | null {
    const config = DATASET_POSE_CONFIG[this.key];
    const index = JOINTS.indexOf(config.joint);
    const value = angles[index] ?? NaN;

    if (!Number.isFinite(value) || value <= 0) return null;

    const zone =
      value <= config.low ? "low" :
      value >= config.high ? "high" : "mid";

    if (this.start === null) {
      if (zone !== "mid") {
        this.start = zone;
        this.reset(value);
      }
      return null;
    }

    this.min = Math.min(this.min, value);
    this.max = Math.max(this.max, value);
    this.frames++;

    const opposite = this.start === "low" ? "high" : "low";
    if (zone === opposite) this.crossed = true;

    if (zone === this.start && this.crossed && this.frames >= 8) {
      const rom = Math.max(
        0,
        Math.min(1, (this.max - this.min) / (config.high - config.low))
      );

      this.reps++;
      this.lastRom = rom;
      this.reset(value);
      return { rom, form: 0.5 };
    }

    return null;
  }

  private reset(value: number) {
    this.crossed = false;
    this.min = value;
    this.max = value;
    this.frames = 0;
  }
}

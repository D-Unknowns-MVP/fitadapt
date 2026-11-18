
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# ==========================================
# 1. FILE LOCATIONS AND EXERCISE SETTINGS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_NAME = input("Enter video filename (e.g. situp.mp4): ")
EXERCISE_NAME = input("Enter exercise name (e.g. Sit Up): ")

VIDEO_PATH = os.path.join(BASE_DIR, "videos", VIDEO_NAME)
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "output",
    os.path.splitext(VIDEO_NAME)[0] + "_angles.csv"
)

LABEL = EXERCISE_NAME

# Automatically create output folder
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ==========================================
# 2. CHECK FILES
# ==========================================

if not os.path.exists(VIDEO_PATH):
    print("ERROR: Video file not found!")
    print("Expected location:", VIDEO_PATH)
    raise SystemExit

if not os.path.exists(MODEL_PATH):
    print("ERROR: Pose model not found!")
    print("Expected location:", MODEL_PATH)
    raise SystemExit

# ==========================================
# 3. JOINT ANGLE CALCULATION
# ==========================================

def calculate_angle(a, b, c):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(np.degrees(radians))

    if angle > 180:
        angle = 360 - angle

    return angle


# ==========================================
# 4. SIGNED GROUND ANGLE CALCULATION
# ==========================================

def calculate_ground_angle(a, b):

    dx = b[0] - a[0]
    dy = b[1] - a[1]

    angle = np.degrees(
        np.arctan2(-dy, dx)
    )

    return angle


# ==========================================
# 5. MEDIAPIPE CONFIGURATION
# ==========================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

# ==========================================
# 6. LEFT BODY LANDMARK IDS
# ==========================================

LEFT_POINTS = {
    "shoulder": 11,
    "elbow": 13,
    "wrist": 15,
    "hip": 23,
    "knee": 25,
    "ankle": 27,
    "foot": 31
}

# ==========================================
# 7. OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video.")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

frame_number = 0
rows = []

pose_detected_frames = 0
rejected_frames = 0

print("Processing video...")
print("Video:", VIDEO_PATH)
print("Exercise:", EXERCISE_NAME)

# ==========================================
# 8. PROCESS VIDEO
# ==========================================

WINDOW_NAME = EXERCISE_NAME + " Video Processing"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    960,
    540
)

with PoseLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Calculate video timestamp
        timestamp_ms = int(
            frame_number * 1000 / fps
        )

        # Detect body landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.pose_landmarks:

            pose_detected_frames += 1

            points = result.pose_landmarks[0]

            # ----------------------------------
            # CHECK LEFT SIDE VISIBILITY
            # ----------------------------------

            left_visibility = np.mean([
                points[i].visibility or 0
                for i in LEFT_POINTS.values()
            ])

            if left_visibility >= 0.2:

                side = "left"

                def get_point(index):

                    return [
                        points[index].x,
                        points[index].y
                    ]

                shoulder = get_point(
                    LEFT_POINTS["shoulder"]
                )

                elbow = get_point(
                    LEFT_POINTS["elbow"]
                )

                wrist = get_point(
                    LEFT_POINTS["wrist"]
                )

                hip = get_point(
                    LEFT_POINTS["hip"]
                )

                knee = get_point(
                    LEFT_POINTS["knee"]
                )

                ankle = get_point(
                    LEFT_POINTS["ankle"]
                )

                foot = get_point(
                    LEFT_POINTS["foot"]
                )

                # ==================================
                # 9. JOINT ANGLES
                # ==================================

                shoulder_angle = calculate_angle(
                    elbow, shoulder, hip
                )

                elbow_angle = calculate_angle(
                    shoulder, elbow, wrist
                )

                hip_angle = calculate_angle(
                    shoulder, hip, knee
                )

                knee_angle = calculate_angle(
                    hip, knee, ankle
                )

                ankle_angle = calculate_angle(
                    knee, ankle, foot
                )

                # ==================================
                # 10. GROUND ANGLES
                # ==================================

                shoulder_ground = calculate_ground_angle(
                    hip, shoulder
                )

                elbow_ground = calculate_ground_angle(
                    wrist, elbow
                )

                hip_ground = calculate_ground_angle(
                    knee, hip
                )

                knee_ground = calculate_ground_angle(
                    ankle, knee
                )

                ankle_ground = calculate_ground_angle(
                    foot, ankle
                )

                # ==================================
                # 11. SAVE ROW
                # ==================================

                row = [
                    side,
                    round(shoulder_angle, 4),
                    round(elbow_angle, 4),
                    round(hip_angle, 4),
                    round(knee_angle, 4),
                    round(ankle_angle, 4),
                    round(shoulder_ground, 4),
                    round(elbow_ground, 4),
                    round(hip_ground, 4),
                    round(knee_ground, 4),
                    round(ankle_ground, 4),
                    LABEL
                ]

                rows.append(row)

                # ----------------------------------
                # DISPLAY DETECTION
                # ----------------------------------

                cv2.putText(
                    frame,
                    EXERCISE_NAME + " Detection - " + side,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # Draw detected landmarks
                for point in points:

                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

            else:

                rejected_frames += 1

        # ----------------------------------
        # SHOW VIDEO
        # ----------------------------------

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        delay = max(1, int(1000 / fps))

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

        frame_number += 1

# ==========================================
# 12. SAVE CSV
# ==========================================

cap.release()
cv2.destroyAllWindows()

columns = [
    "Side",
    "Shoulder_Angle",
    "Elbow_Angle",
    "Hip_Angle",
    "Knee_Angle",
    "Ankle_Angle",
    "Shoulder_Ground_Angle",
    "Elbow_Ground_Angle",
    "Hip_Ground_Angle",
    "Knee_Ground_Angle",
    "Ankle_Ground_Angle",
    "Label"
]

with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(columns)
    writer.writerows(rows)

# ==========================================
# 13. FINAL RESULTS
# ==========================================

print("--------------------------------")
print("VIDEO PROCESSING COMPLETE")
print("Total frames processed:", frame_number)
print("Frames with detected poses:", pose_detected_frames)
print("Rejected frames:", rejected_frames)
print("Valid pose records:", len(rows))
print("CSV saved as:", OUTPUT_CSV)
print("--------------------------------")

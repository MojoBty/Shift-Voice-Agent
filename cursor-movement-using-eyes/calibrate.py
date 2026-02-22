#!/usr/bin/env python3
"""
Standalone calibration for eye tracking.
Runs best-gaze static + dynamic calibration and saves calibration.json.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_GAZE_DIR = os.path.join(SCRIPT_DIR, "best-gaze")
sys.path.insert(0, BEST_GAZE_DIR)

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# Import calibration routines from best-gaze
import main as bg

DIR = BEST_GAZE_DIR
MODEL_PATH = os.path.join(DIR, "face_landmarker.task")
CALIB_PATH = os.path.join(DIR, "calibration.json")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: face_landmarker.task not found at {MODEL_PATH}")
        sys.exit(1)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Cannot open webcam")
        sys.exit(1)

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    lmk = vision.FaceLandmarker.create_from_options(opts)
    t0 = __import__("time").time()

    print("\n=== STATIC CALIBRATION ===\n")
    print("Look at each dot and press SPACE. Press Q to quit.\n")
    cal = bg.run_calibration(lmk, cam, t0)
    if not cal:
        lmk.close()
        cam.release()
        sys.exit(1)

    print("\n=== DYNAMIC CALIBRATION ===\n")
    print("Follow the moving dot with your eyes for ~15 seconds. Press Q to skip.\n")
    cal = bg.run_dynamic_calibration(lmk, cam, t0, cal)

    lmk.close()
    cam.release()
    cv2.destroyAllWindows()
    print("\nCalibration complete! You can now use the extension.")


if __name__ == "__main__":
    main()

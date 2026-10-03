"""
make_test_video.py

Generates a SYNTHETIC test video used only to self-test the pipeline in
this environment (no real race footage was provided). It renders a
moving "car" (a rectangle with wheel-like corner markers) drifting
across a track with a white boundary line, so the vehicle detector's
classical-CV fallback, the boundary detector, wheel estimation, and
violation logic can all be exercised on REAL pixel data end-to-end.

This is clearly a stand-in for actual race footage — for the hackathon
submission, replace this with genuine race video and (ideally) install
`ultralytics` so VehicleDetector uses real pretrained YOLO detection.
"""

import cv2
import numpy as np
import os

OUT_PATH = "sample_videos/synthetic_test_track.mp4"
WIDTH, HEIGHT = 960, 540
FPS = 30
DURATION_SEC = 8

TRACK_COLOR = (60, 90, 60)          # dark green "grass"/track surface
ASPHALT_COLOR = (70, 70, 70)
LINE_Y_START = 360                   # boundary line vertical position (roughly horizontal line)
LINE_THICKNESS = 6
CAR_COLOR = (40, 90, 200)
CAR_W, CAR_H = 90, 45


def draw_scene(frame, car_cx, car_cy, line_y):
    frame[:, :] = TRACK_COLOR
    cv2.rectangle(frame, (0, 0), (WIDTH, line_y - 40), ASPHALT_COLOR, -1)
    # white boundary line (slightly wavy-free, straight for this synthetic clip)
    cv2.line(frame, (0, line_y), (WIDTH, line_y), (245, 245, 245), LINE_THICKNESS)

    x1 = int(car_cx - CAR_W / 2)
    y1 = int(car_cy - CAR_H / 2)
    x2 = int(car_cx + CAR_W / 2)
    y2 = int(car_cy + CAR_H / 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), CAR_COLOR, -1)
    # simple "wheels" so the shape reads as a vehicle
    for wx in (x1 + 8, x2 - 8):
        for wy in (y1 + 6, y2 - 6):
            cv2.circle(frame, (wx, wy), 6, (15, 15, 15), -1)
    return frame


def main():
    os.makedirs("sample_videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    total_frames = FPS * DURATION_SEC
    car_cy = LINE_Y_START - 60  # starts safely on-track (above the line)

    for i in range(total_frames):
        t = i / total_frames
        # Car moves left->right along the straight, and drifts DOWN
        # (across the line) during the middle third, then recovers.
        car_cx = 80 + t * (WIDTH - 160)

        if t < 0.35:
            drift = 0
        elif t < 0.65:
            # smoothly drift down across the line and fully beyond it
            local_t = (t - 0.35) / 0.30
            drift = local_t * 90
        else:
            # recover back onto track
            local_t = (t - 0.65) / 0.35
            drift = max(0, 90 * (1 - local_t))

        cy = car_cy + drift

        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame = draw_scene(frame, car_cx, cy, LINE_Y_START)
        writer.write(frame)

    writer.release()
    print(f"Wrote synthetic test video: {OUT_PATH} ({total_frames} frames @ {FPS}fps)")


if __name__ == "__main__":
    main()

"""Overhead camera controller: detect parts (bottle, clock, cat) with YOLO and broadcast them.

Pipeline (every INFER_EVERY_N_STEPS sim steps):
  1. Grab the camera image (BGRA bytes) -> numpy BGR array.
  2. Run a COCO-pretrained YOLO11n model, keeping only the classes in TARGET_CLASSES.
  3. Convert each box center from pixels to world (x, y) with the pinhole model.
  4. Send one JSON message per detection over the Emitter (channel 1):
       {"t": sim_time, "class": "bottle", "conf": 0.87, "x": -0.80, "y": 0.10}

Note: the camera's Recognition node is NOT used here. Detection is YOLO only.
"""

import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from controller import Robot

# ----------------------------------------------------------------------------
# Settings (change these, not the code below)
# ----------------------------------------------------------------------------
INFER_EVERY_N_STEPS = 4      # YOLO is slow relative to the 16 ms sim step
CONF_THRESHOLD = 0.25        # minimum YOLO confidence to keep a detection
PRINT_ALL_CLASSES = True     # debug: also print non-fruit detections (not sent)
SAVE_FIRST_N_FRAMES = 3      # save this many frames (raw + annotated) to ./frames

# COCO class ids in the pretrained model -> the names we send to the arm
TARGET_CLASSES = {15: "cat", 39: "bottle", 74: "clock"}

# Camera pose in the world. Must match DEF OVERHEAD_CAMERA in conveyor_ur5e.wbt.
# (A plain Robot cannot read its own pose, so it is written down here.)
CAMERA_X, CAMERA_Y, CAMERA_Z = -1.0, 0.0, 1.6

# Height of each object's CENTER above the belt, used for the pixel->world
# scale. The box drawn by YOLO includes the object's visible sides, so its
# center lines up best with the object's mid-height, not its top.
BELT_TOP_Z = 0.60
CENTER_HEIGHT_ABOVE_BELT = {
    "bottle": 0.044,   # lying on its side, radius ~4.4 cm
    "cat": 0.0226,        # lying on its side, 4.5 cm thick
    "clock": 0.010,       # lying face up, 2 cm thick
}

CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(CONTROLLER_DIR, "yolo11n.pt")   # downloaded on first run
FRAMES_DIR = os.path.join(CONTROLLER_DIR, "frames")


def camera_to_bgr(camera):
    """Webots gives BGRA bytes, row-major. Reshape to (H, W, 4) and drop alpha."""
    h, w = camera.getHeight(), camera.getWidth()
    bgra = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((h, w, 4))
    return np.ascontiguousarray(bgra[:, :, :3])


def pixel_to_world(u, v, img_w, img_h, fov, object_center_z):
    """Pinhole model for a camera looking straight down.

    u = pixel column (0 at left), v = pixel row (0 at top).
    object_center_z = world height of the object's center (sets the scale).

    Scale: at distance d below the camera, the image spans 2*d*tan(fov/2)
    meters horizontally, so one pixel = that / image width. (fov is the
    horizontal field of view; pixels are square, so the same scale works
    vertically.)

    Axes: the Webots camera looks along its local +X, with local +Z = image up
    and local +Y = image left. The camera is rotated 90 deg about world Y
    (rotation 0 1 0 1.5708), which turns local +X to world -Z (looking down),
    local +Z to world +X, and local +Y stays world +Y. So:
        image up    (v decreasing) -> world +X  (the belt's travel direction)
        image left  (u decreasing) -> world +Y
    """
    d = CAMERA_Z - object_center_z
    meters_per_pixel = 2.0 * d * math.tan(fov / 2.0) / img_w
    x = CAMERA_X + (img_h / 2.0 - v) * meters_per_pixel
    y = CAMERA_Y + (img_w / 2.0 - u) * meters_per_pixel
    return x, y


def main():
    print(f"[vision] python {sys.version.split()[0]} ({sys.executable})", flush=True)
    print(f"[vision] ultralytics {ultralytics.__version__}, torch {torch.__version__}", flush=True)

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)
    emitter = robot.getDevice("emitter")

    img_w, img_h, fov = camera.getWidth(), camera.getHeight(), camera.getFov()
    print(f"[vision] camera {img_w}x{img_h}, fov {fov:.3f} rad", flush=True)

    # Load the model once. Use the Apple GPU (mps) if available.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(WEIGHTS)
    model.predict(np.zeros((img_h, img_w, 3), np.uint8), device=device, verbose=False)  # warm-up
    print(f"[vision] loaded {os.path.basename(WEIGHTS)} on {device}", flush=True)

    if SAVE_FIRST_N_FRAMES > 0:
        os.makedirs(FRAMES_DIR, exist_ok=True)
    frames_saved = 0
    step = 0

    while robot.step(timestep) != -1:
        step += 1
        if step % INFER_EVERY_N_STEPS != 0:
            continue

        t_sim = robot.getTime()
        frame = camera_to_bgr(camera)

        t0 = time.perf_counter()
        result = model.predict(frame, conf=CONF_THRESHOLD, device=device, verbose=False)[0]
        infer_ms = (time.perf_counter() - t0) * 1000

        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            u, v = box.xywh[0][:2].tolist()   # box center in pixels

            if cls_id not in TARGET_CLASSES:
                if PRINT_ALL_CLASSES:
                    print(f"[vision] t={t_sim:.2f} ignored '{model.names[cls_id]}' "
                          f"conf={conf:.2f} at pixel ({u:.0f},{v:.0f})", flush=True)
                continue

            name = TARGET_CLASSES[cls_id]
            z_center = BELT_TOP_Z + CENTER_HEIGHT_ABOVE_BELT[name]
            x, y = pixel_to_world(u, v, img_w, img_h, fov, z_center)
            msg = {"t": round(t_sim, 3), "class": name,
                   "conf": round(conf, 3), "x": round(x, 4), "y": round(y, 4)}
            emitter.send(json.dumps(msg))
            print(f"[vision] {json.dumps(msg)}  pixel=({u:.0f},{v:.0f})  "
                  f"infer={infer_ms:.0f}ms", flush=True)

        if frames_saved < SAVE_FIRST_N_FRAMES:
            cv2.imwrite(os.path.join(FRAMES_DIR, f"frame_{frames_saved}_raw.png"), frame)
            cv2.imwrite(os.path.join(FRAMES_DIR, f"frame_{frames_saved}_yolo.png"), result.plot())
            frames_saved += 1


main()

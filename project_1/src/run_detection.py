#!/usr/bin/env python3
"""
Run live object detection on a camera feed using a YOLO model.
"""
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

MODEL = "records_books_best.pt"  # weights filename (looked up in weights/) or a full path
CONF = 0.25  # confidence threshold
IOU = 0.45  # NMS IoU threshold
CAMERA = 1  # camera index (use 0 if that's your phone via Continuity Camera)
SOURCE = None  # video file path, overrides CAMERA if set
CLASSES = None  # restrict to these class ids, e.g. [73] for COCO "book"; None for all
WARMUP_FRAMES = 15  # frames to discard while exposure settles
DEVICE = None  # override auto device selection (mps/cuda/cpu); None = auto
LIST_CAMERAS_ONLY = False  # if True, just print available camera indexes and exit

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def list_cameras(max_index=5):
    """List available camera indexes and whether they can be opened and read from."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        opened = cap.isOpened()
        frame_ok = cap.read()[0] if opened else False
        cap.release()
        if not opened:
            break
        print(f"index {i}: opened={opened} frame_ok={frame_ok}")


def pick_device() -> str:
    """Return 'mps', 'cuda', or 'cpu', in that priority order."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def open_capture(source, warmup_frames=WARMUP_FRAMES) -> cv2.VideoCapture:
    """Open the camera to be used for live detection."""
    is_video_file = isinstance(source, str) and Path(source).exists()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {source!r}")
    if not is_video_file:
        for _ in range(warmup_frames):
            cap.read()
    return cap


def main():
    if LIST_CAMERAS_ONLY:
        list_cameras()
        return

    device = DEVICE or pick_device()
    print(f"Using device: {device}")

    # Load the model and open the camera feed
    weights_path = MODEL if Path(MODEL).exists() else str(WEIGHTS_DIR / MODEL)
    model = YOLO(weights_path)

    source = SOURCE if SOURCE else CAMERA
    cap = open_capture(source, warmup_frames=WARMUP_FRAMES)

    last_print = time.time()

    print("Press Ctrl-C in the terminal to quit")

    # Run detection on each frame until the camera stops or Ctrl-C is pressed
    frame_count = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                results = model(frame, conf=CONF, iou=IOU, classes=CLASSES, device=device, verbose=False)
                annotated_frame = results[0].plot()

                cv2.putText(
                    annotated_frame,
                    f"device: {device}  model: {MODEL}  conf: {CONF}  iou: {IOU}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("YOLO Inference", annotated_frame)
                frame_count += 1

                if time.time() - last_print > 2.0:
                    print(f"detections: {len(results[0].boxes)}")
                    last_print = time.time()

                cv2.waitKey(1)
            else:
                print("Frame grab failed (or video file ended), stopping.")
                break
    except KeyboardInterrupt:
        print("\nStopped (Ctrl-C).")

    print(f"Ran {frame_count} frames.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

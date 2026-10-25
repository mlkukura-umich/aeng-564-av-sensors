# AENG 564 - Project 1: Records vs. Books Detector

This project fine-tunes a YOLO11 model to tell vinyl records and books
apart on a live camera feed. The stock, pretrained model cannot do this,
since it only recognizes "book" from COCO and has no concept of "record"
at all.

## 1. Clone and set up the environment

```bash
git clone <repo-url>
cd project_1
python -m venv .venv
```

Activate the virtual environment:
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Windows (cmd): `.venv\Scripts\activate.bat`
- macOS / Linux: `source .venv/bin/activate`

Install dependencies:
```bash
pip install -r requirements.txt
```

This was developed and tested on macOS, but the code itself can run on Windows and Linux as well. The compute device
is also selected automatically at runtime, so nothing needs to be
configured manually there either.

## 2. Find your camera index

Camera index is not predictable across machines, so it should be checked
first rather than assumed. In `run_detection.py`, set
`LIST_CAMERAS_ONLY = True` near the top, then run:

```bash
cd src
python run_detection.py
```

This prints which camera indexes open and successfully grab a frame. Note
which index corresponds to the camera to be used (built-in webcam,
external camera, or a phone connected as a webcam), then set
`LIST_CAMERAS_ONLY` back to `False`.

## 3. Run live detection

Settings are set as
constants near the top of each file. Open `run_detection.py` and set
`CAMERA` to the index found in step 2, then run:

```bash
cd src
python run_detection.py
```

This opens a window with the live feed and detection boxes. The video window can be closed by pressing Ctrl-C in the
terminal.

A few other settings worth trying can also be changed at the top of
`run_detection.py`:

```python
# Compare against the original, non-fine-tuned model (filtered to only the "book" class)
MODEL = "yolo11n.pt"
CLASSES = [73]

# Adjust detection confidence/overlap thresholds
CONF = 0.5
IOU = 0.45
```

#!/usr/bin/env python3
"""
Fine-tune YOLO11 on the records-vs-books dataset.
"""
import tempfile
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

MODEL = "yolo11n.pt"
EPOCHS = 75
PATIENCE = 20
IMGSZ = 640
BATCH = 16
DEVICE = None  # override auto device selection (mps/cuda/cpu); None = auto
SEED = 42
NAME = None  # run name under runs/; None = derived from MODEL

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DATA_YAML = Path(__file__).resolve().parent.parent / "data" / "record_book.yaml"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def pick_device() -> str:
    """Return 'mps', 'cuda', or 'cpu', in that priority order."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_data_yaml() -> str:
    """
    Point the data yaml to the labeled data directory, and return the path to this config file.
    """
    with open(DATA_YAML) as f:
        config = yaml.safe_load(f)
    config["path"] = str(DATA_YAML.parent / "labeled")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(config, tmp)
    tmp.close()
    return tmp.name


def main():

    # Load the base model and fine-tune it on the labeled dataset
    weights_path = WEIGHTS_DIR / MODEL
    model = YOLO(str(weights_path) if weights_path.exists() else MODEL)

    run_name = NAME or f"{Path(MODEL).stem}_records_books"

    results = model.train(
        data=resolve_data_yaml(),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE or pick_device(),
        seed=SEED,
        deterministic=True,
        name=run_name,
        project=str(RUNS_DIR),
    )

    # Report where the trained weights and training plots ended up
    print("Training complete.")
    print(f"Results saved under {RUNS_DIR / run_name}")


if __name__ == "__main__":
    main()

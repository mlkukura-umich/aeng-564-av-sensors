#!/usr/bin/env python3
"""
Evaluate the trained model on the test set and save a summary of metrics.
"""
import json
import tempfile
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

WEIGHTS = "../weights/records_books_best.pt"
DEVICE = None  # override auto device selection (mps/cuda/cpu); None = auto

DATA_YAML = Path(__file__).resolve().parent.parent / "data" / "record_book.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


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

    # Load the model and evaluate on the test set
    model = YOLO(WEIGHTS)

    metrics = model.val(
        data=resolve_data_yaml(),
        split="test",
        device=DEVICE or pick_device(),
        project=str(RESULTS_DIR / "eval"),
        name="test_set",
    )

    # Save a summary of metrics to a JSON file
    names = metrics.names
    per_class_ap50 = metrics.box.ap50
    per_class_ap = metrics.box.ap

    summary = {
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
        "per_class": {},
    }
    for i, name in names.items():
        summary["per_class"][name] = {
            "AP50": float(per_class_ap50[i]) if i < len(per_class_ap50) else None,
            "AP50-95": float(per_class_ap[i]) if i < len(per_class_ap) else None,
            "ground_truth_instances": int(metrics.nt_per_class[i]) if metrics.nt_per_class is not None else None,
        }

    out_path = RESULTS_DIR / "evaluation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {out_path}")
    print(f"Plots saved under {RESULTS_DIR / 'eval' / 'test_set'}")


if __name__ == "__main__":
    main()

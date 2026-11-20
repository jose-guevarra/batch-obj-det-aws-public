from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COCO = ROOT / "data/datasets/coco"
IMAGES = COCO / "val2017"
ANNOTATIONS = COCO / "annotations/instances_val2017.json"
WEIGHTS = ROOT / "models/yolov8m.pt"


@pytest.fixture(scope="session")
def coco_files() -> tuple[Path, Path]:
    if not IMAGES.is_dir() or not ANNOTATIONS.is_file():
        pytest.skip(f"COCO val2017 not found under {COCO} (see data/dl_data.sh)")
    return IMAGES, ANNOTATIONS


@pytest.fixture(scope="session")
def weights() -> Path:
    if not WEIGHTS.is_file():
        pytest.skip(f"{WEIGHTS} missing: run scripts/evaluate_map.py once to download it")
    return WEIGHTS


@pytest.fixture(scope="session")
def torch_runtime(weights):
    from detector.runtimes import TorchRuntime

    return TorchRuntime(weights, device="auto")

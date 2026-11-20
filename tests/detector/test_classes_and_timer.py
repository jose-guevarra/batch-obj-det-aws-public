import time

from detector.inference_core import COCO80_TO_91, COCO_NAMES, StageTimer


def test_coco_class_mapping_matches_ultralytics():
    from ultralytics.data.converter import coco80_to_coco91_class

    assert len(COCO80_TO_91) == 80
    assert len(set(COCO80_TO_91)) == 80
    assert list(COCO80_TO_91) == list(coco80_to_coco91_class())
    assert (COCO80_TO_91[0], COCO80_TO_91[79]) == (1, 90)


def test_coco_names_match_ultralytics_dataset_config():
    from ultralytics.utils import ROOT, YAML

    names = YAML.load(ROOT / "cfg/datasets/coco.yaml")["names"]
    assert tuple(names[i] for i in range(80)) == COCO_NAMES


def test_timer_accumulates_and_syncs_before_stopping():
    calls = []
    timer = StageTimer(sync=lambda: calls.append("sync"))
    for _ in range(3):
        with timer.stage("inference"):
            time.sleep(0.002)
    assert timer.counts["inference"] == 3
    assert calls == ["sync"] * 3
    summary = timer.summary(batches=3)
    assert summary["inference"]["total_s"] >= 0.006
    assert summary["inference"]["per_batch_ms"] >= 2

import json

from detector.evaluation import coco_detections, part_lines_to_coco


def line(name, status="done", dets=()):
    record = {"batch_id": "b", "image_name": name, "status": status}
    if status == "done":
        record["detections"] = [{"box": b, "score": s, "class_id": c} for b, s, c in dets]
    else:
        record["error"] = "decode failed: not a decodable image"
    return json.dumps(record)


def test_part_lines_to_coco():
    lines = [
        line(
            "000000000139.jpg", dets=[([10.0, 20.0, 110.0, 70.0], 0.9, 0), ([0, 0, 5, 5], 0.2, 79)]
        ),
        line("000000000285.jpg"),  # no detections, but still scored
        "",
        line("corrupt-000.jpg", status="failed"),
    ]
    dets, image_ids, failed = part_lines_to_coco(lines)

    assert image_ids == [139, 285]
    assert failed == ["corrupt-000.jpg"]
    assert dets == [
        {"image_id": 139, "category_id": 1, "bbox": [10.0, 20.0, 100.0, 50.0], "score": 0.9},
        {"image_id": 139, "category_id": 90, "bbox": [0.0, 0.0, 5.0, 5.0], "score": 0.2},
    ]  # class 0 (person) -> 1, class 79 (toothbrush) -> 90; xyxy -> xywh


def test_coco_detections_rounding():
    (d,) = coco_detections(7, [((1.234, 2.345, 3.456, 4.567), 0.123456, 2)])
    assert d == {
        "image_id": 7,
        "category_id": 3,
        "bbox": [1.23, 2.35, 2.22, 2.22],
        "score": 0.12346,
    }

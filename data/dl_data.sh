#!/bin/bash

mkdir -p datasets/coco && cd datasets/coco
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip

# for annotations, you only need instances_val2017.json (mAP is scored locally,
# see scripts/evaluate_map.py; the annotations never go to S3)

# To process images on AWS, upload them as a batch (writes the batch record first,
# then copies to s3://<bucket>/input/<batch_id>/ with s5cmd):
#   uv run python scripts/upload_batch.py --limit 100 --wait
#   (or from infra/: make upload N=100)
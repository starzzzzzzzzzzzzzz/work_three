#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m scripts.patch_artifact_probe \
  --model-path /Volumes/mydatabase/LM/models/llava-1.5-7b-hf \
  --image-dir /Volumes/mydatabase/data/datasets/coco/Image/val2014 \
  --instances /Volumes/mydatabase/data/datasets/coco/Annotations/annotations/instances_val2014.json \
  --output-dir runs/patch_artifact_probe_500 \
  --num-images 500 \
  --seed 7 \
  --topk 50 \
  --min-area-ratio 0.05 \
  --max-area-ratio 0.65 \
  --max-annotations 2 \
  --device auto


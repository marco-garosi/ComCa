#! /bin/bash

base_path="/mnt/data/ComCa"
cache_path="/mnt/data/.cache"

python scripts/run_inference_detection.py \
    --model_from OpenCLIP \
    --model_arch "ViT-B-32" \
    --pretrained "laion2b_e16" \
    --image_size 224 \
    -bs 32 \
    --template_file OvarNet.json \
    --prompt none \
    --path_for_image_features "$base_path/OVAD/ovad-benchmark-code/" \
    --output_dir "$base_path/OVAD/ovad-benchmark-code/output/OVAD_Benchmark2000/multimodal_ovac/open_clip_ours/" \
    --dir_data "$base_path/OVAD/ovad-benchmark-code/datasets/coco" \
    --model_cache_dir "$cache_path" \
    --no-use_cache \
    --detection_model "yolo11n.pt"

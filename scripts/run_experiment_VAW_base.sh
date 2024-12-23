#! /bin/bash

base_path="/mnt/data/ComCa"
cache_path="/mnt/data/.cache"

python scripts/run_inference_VAW.py \
    --model_from OpenCLIP \
    --model_arch "ViT-B-32" \
    --pretrained "laion2b_e16" \
    --image_size 224 \
    -bs 256 \
    --template_file OvarNet.json \
    --prompt none \
    --no-cache_by_category \
    --no-cache_by_sample \
    --scale_base_logits 1.0 \
    --scale_logits none \
    --softmax none \
    --path_for_image_features "$base_path/OVAD/ovad-benchmark-code/" \
    --path_to_cache "$base_path/comca-cache/VAW" \
    --output_dir "$base_path/OVAD/ovad-benchmark-code/output/VAW/multimodal_ovac/open_clip_ours/" \
    --dir_data "$base_path/VAW" \
    --no-store_predictions \
    --model_cache_dir "$cache_path" \
    --no-use_cache

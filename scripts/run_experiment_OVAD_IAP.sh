#! /bin/bash

base_path="/mnt/data/ComCa"
cache_path="/mnt/data/.cache"

python scripts/run_inference_TIP-IAP.py \
    --model_from OpenCLIP \
    --model_arch "ViT-B-32" \
    --pretrained laion2b_e16 \
    -bs 100 \
    --template_file ovad.json \
    --prompt all \
    --prefix categories_ \
    --shots 16 \
    --seed 0 \
    --soft_labels "" \
    --no-cache_by_category \
    --alpha 1.17 \
    --beta 1.0 \
    --path_for_image_features "$base_path/OVAD/ovad-benchmark-code/" \
    --path_to_cache "$base_path/comca-cache/OVAD" \
    --output_dir "$base_path/OVAD/ovad-benchmark-code/output/OVAD_Benchmark2000/multimodal_ovac/open_clip_ours/" \
    --dir_data "$base_path/OVAD/ovad-benchmark-code/datasets/ovad_box_instances/2000_img" \
    --no-store_predictions \
    --model_cache_dir "$cache_path"

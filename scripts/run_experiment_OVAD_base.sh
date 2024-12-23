#! /bin/bash

base_path="/mnt/data/ComCa"
cache_path="/mnt/data/.cache"

python scripts/run_inference.py \
    --model_from OpenCLIP \
    --model_arch "ViT-B-32" \
    --pretrained "laion2b_e16" \
    --image_size 224 \
    -bs 256 \
    --template_file OvarNet.json \
    --prompt none \
    --path_for_image_features "$base_path/OVAD/ovad-benchmark-code/" \
    --output_dir "$base_path/OVAD/ovad-benchmark-code/output/OVAD_Benchmark2000/multimodal_ovac/open_clip_ours/" \
    --dir_data "$base_path/OVAD/ovad-benchmark-code/datasets/ovad_box_instances/2000_img" \
    --no-use_cache

# To reproduce results from OVAD, use `--template_file ovad.json` and `--prompt a`
# To use the same prompts employed in all experiments, use `--template_file OvarNet.json`` and  `--prompt none`
# These prompts are derived from OvarNet and are the same for all attributes, regardless
# of them being "is" or "has" type attributes, which is a specific annotation in the OVAD
# benchmark.

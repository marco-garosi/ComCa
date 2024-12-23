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
    --prefix retrieved_from_cc12m_faiss_t2i_none_sampled_from_cc12m-1_reweighted_with_gpt-3.5-turbo-0125_24_top-0.8_weighted \
    --shots 16 \
    --seed 0 \
    --no-cache_by_category \
    --no-cache_by_sample \
    --scale_base_logits 1.0 \
    --scale_logits "max" \
    --softmax "overall" \
    --alpha 1.17 \
    --beta 1.0 \
    --path_for_image_features "$base_path/OVAD/ovad-benchmark-code/" \
    --path_to_cache "$base_path/ovad-cache/OVAD" \
    --output_dir "$base_path/OVAD/ovad-benchmark-code/output/OVAD_Benchmark2000/multimodal_ovac/open_clip_ours/" \
    --dir_data "$base_path/OVAD/ovad-benchmark-code/datasets/ovad_box_instances/2000_img" \
    --no-store_predictions \
    --model_cache_dir "$cache_path" \
    --soft_labels "alpha_0.6_given_mean_and_std_by_object"

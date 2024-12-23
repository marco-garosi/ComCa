#! /bin/bash

base_path="/mnt/data/ComCa"
cache_path="/mnt/data/.cache"

python scripts/run_inference_VAW_SuS-X.py \
    --model_from OpenCLIP \
    --model_arch "ViT-B-32" \
    --pretrained "laion2b_e16" \
    --image_size 224 \
    -bs 32 \
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
    --path_to_cache "$base_path/comca-cache/VAW" \
    --output_dir "$base_path/OVAD/ovad-benchmark-code/output/VAW/multimodal_ovac/open_clip_ours/" \
    --dir_data "$base_path/VAW" \
    --no-store_predictions \
    --model_cache_dir "$cache_path" #\
    # --soft_labels "alpha_0.6_given_mean_and_std_by_object_s"


# Use the first cache to run with plain SuS-X,
# the second one to use our sampling strategy to build the cache.
# --prefix retrieved_from_cc12m_faiss_t2i_none_sampled_from_cc12m-1_random_uniform \
# --prefix retrieved_from_cc12m_faiss_t2i_none_sampled_from_cc12m-1_reweighted_with_gpt-3.5-turbo-0125_24_top-0.8_weighted \
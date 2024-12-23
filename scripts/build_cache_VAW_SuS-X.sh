#! /bin/bash

base_path="/mnt/data/ComCa"
cache_path="/mnt/data/.cache"

python scripts/build_cache.py \
    -d VAW \
    --distribution 'cc12m-1' \
    --mode 'random_uniform' \
    --retrieval_modality image \
    --prompt_bs 512 \
    --seed 0 \
    --alpha 0.6 \
    --template_file OvarNet_with_category.json \
    --template none

#!/bin/bash
#SBATCH -p partition
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8196MB
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH --output=scripts/results.txt

module load cuda

source /home/${USER}/.bashrc
conda activate image_generation

accelerate launch scripts/generate_cache_all_pairs.py --model=stabilityai/stable-diffusion-xl-base-1.0 --scheduler=dpm_ms --num_inference_steps=20 --epochs=16 --bs=4 --templates_file=001.json --template=none --negative_prompts --out_folder=sd_all_pairs_001 --resume_from_epoch=0
#!/bin/bash
#SBATCH --job-name=debug_mmed
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --output=debug_mmed_%j.out
#SBATCH --error=debug_mmed_%j.err

source /home/gkianfar/scratch/Amin/conceptvenv/bin/activate
cd /home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis

# Run ONE image only — no /dev/null redirect
python run_x_to_c_to_y.py \
    --dataset PH2 --split 0 \
    --model Explicd --concept_extractor Explicd \
    --generate_concepts \
    --data_path data \
    --refiner mmed

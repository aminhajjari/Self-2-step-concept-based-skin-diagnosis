#!/bin/bash
#SBATCH --job-name=medgemma_cy
#SBATCH --account=def-arashmoh_gpu
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/gkianfar/scratch/Amin/concept/outputs/logs/medgemma_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/concept/outputs/logs/medgemma_%j.err

set -e
module load gcc python/3.11.5 opencv/4.13.0
source /home/gkianfar/scratch/Amin/conceptvenv311/bin/activate

PROJECT_PATH=/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis
MG_CKPT="$PROJECT_PATH/checkpoint/MedGemma-4b-it"
RUN="python $PROJECT_PATH/run_x_to_c_to_y.py"

# results/ resolves here (same dir your other outputs use)
cd /home/gkianfar/scratch/Amin/concept/outputs

# ---------- ZERO-SHOT: all datasets x all refiners ----------
for refiner in rule mistral mmed; do
  for split in 0 1 2 3 4; do
    $RUN --dataset PH2 --split $split --concept_extractor Explicd \
         --llm MedGemma --ckpt $MG_CKPT --n_demos 0 --refiner $refiner
  done
  for dataset in Derm7pt HAM10000; do
    $RUN --dataset $dataset --concept_extractor Explicd \
         --llm MedGemma --ckpt $MG_CKPT --n_demos 0 --refiner $refiner
  done
done

# ---------- FEW-SHOT (RICES): match your existing shot grid ----------
for n_shots in 1 2 4 8; do
  for refiner in rule mistral mmed; do
    for split in 0 1 2 3 4; do
      $RUN --dataset PH2 --split $split --concept_extractor Explicd \
           --llm MedGemma --ckpt $MG_CKPT --use_demos --n_demos $n_shots --refiner $refiner
    done
    $RUN --dataset Derm7pt --concept_extractor Explicd \
         --llm MedGemma --ckpt $MG_CKPT --use_demos --n_demos $n_shots --refiner $refiner
  done
done

# HAM10000 few-shot only goes to 2 in your existing runs
for n_shots in 1 2; do
  for refiner in rule mistral mmed; do
    $RUN --dataset HAM10000 --concept_extractor Explicd \
         --llm MedGemma --ckpt $MG_CKPT --use_demos --n_demos $n_shots --refiner $refiner
  done
done

echo "MEDGEMMA SWEEP DONE"

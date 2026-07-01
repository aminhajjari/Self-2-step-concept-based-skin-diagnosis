#!/bin/bash
#SBATCH --job-name=qwen_cy
#SBATCH --account=def-arashmoh_gpu
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/home/gkianfar/scratch/Amin/concept/outputs/logs/qwen_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/concept/outputs/logs/qwen_%j.err

set -e
module load gcc python/3.11.5 opencv/4.13.0 arrow
source /home/gkianfar/scratch/Amin/conceptvenv311/bin/activate

PROJECT=/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis
CKPT="$PROJECT/checkpoint/Qwen2.5-72B-Instruct"
RUN="python $PROJECT/run_x_to_c_to_y.py"
cd /home/gkianfar/scratch/Amin/concept/outputs

# ---------- ZERO-SHOT: all datasets x all refiners ----------
for refiner in rule mistral mmed; do
  for split in 0 1 2 3 4; do
    $RUN --dataset PH2 --split $split --concept_extractor Explicd \
         --llm Qwen --ckpt $CKPT --n_demos 0 --refiner $refiner
  done
  for dataset in Derm7pt HAM10000; do
    $RUN --dataset $dataset --concept_extractor Explicd \
         --llm Qwen --ckpt $CKPT --n_demos 0 --refiner $refiner
  done
done

# ---------- FEW-SHOT (RICES): PH2 + Derm7pt to 8, HAM to 2 ----------
for n in 1 2 4 8; do
  for refiner in rule mistral mmed; do
    for split in 0 1 2 3 4; do
      $RUN --dataset PH2 --split $split --concept_extractor Explicd \
           --llm Qwen --ckpt $CKPT --use_demos --n_demos $n --refiner $refiner
    done
    $RUN --dataset Derm7pt --concept_extractor Explicd \
         --llm Qwen --ckpt $CKPT --use_demos --n_demos $n --refiner $refiner
  done
done

for n in 1 2; do
  for refiner in rule mistral mmed; do
    $RUN --dataset HAM10000 --concept_extractor Explicd \
         --llm Qwen --ckpt $CKPT --use_demos --n_demos $n --refiner $refiner
  done
done

echo "QWEN SWEEP DONE"

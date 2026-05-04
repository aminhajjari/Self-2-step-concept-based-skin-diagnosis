#!/bin/bash
#SBATCH --job-name=xcy_selfrefine
#SBATCH --account=def-arashmoh
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/gkianfar/scratch/Amin/concept/outputs/logs/xcy_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/concept/outputs/logs/xcy_%j.err

set -e

echo "========================================="
echo "FULL x→c→y PIPELINE WITH SELF-REFINE"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "========================================="

# ==============================
# LOAD MODULES (CRITICAL ORDER)
# ==============================
module purge
module load gcc python/3.10 cuda/12.6 opencv/4.10.0

echo "✓ Modules loaded"

# ==============================
# ACTIVATE ENV (AFTER MODULES)
# ==============================
source /home/gkianfar/scratch/Amin/conceptvenv/bin/activate
echo "✓ Environment: $VIRTUAL_ENV"

# ==============================
# VERIFY OPENCV (DEBUG - KEEP THIS)
# ==============================
python - <<EOF
import cv2
print("✓ OpenCV version:", cv2.__version__)
print("✓ OpenCV path:", cv2.__file__)
EOF

# ==============================
# OFFLINE MODE
# ==============================
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TIMM_FUSED_ATTN=0

# ==============================
# PATHS
# ==============================
PROJECT_PATH="/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis"
DATA_PATH="$PROJECT_PATH/data"
OUTPUT_BASE="/home/gkianfar/scratch/Amin/concept/outputs"
CKPT_PATH="$PROJECT_PATH/checkpoint/MMed-Llama-3-8B"

cd $PROJECT_PATH || exit 1

# ==============================
# OUTPUT STRUCTURE
# ==============================
mkdir -p $OUTPUT_BASE/logs
mkdir -p $OUTPUT_BASE/results/concept_prediction
mkdir -p $OUTPUT_BASE/results/label_prediction

[ -d results ] && rm -rf results
ln -s $OUTPUT_BASE/results results

# ==============================
# GPU INFO
# ==============================
echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==============================
# RUN FUNCTION (avoid repetition)
# ==============================
run_stage () {
    python run_x_to_c_to_y.py "$@" 2>&1 | tee "$LOG_FILE"
    python -c "import torch; torch.cuda.empty_cache()"
    sleep 2
}

# ============================================
# PH2 (5 splits)
# ============================================
echo "========== PH2 =========="

for split in 0 1 2 3 4; do
    echo "---- PH2 Split $split ----"

    LOG_FILE=$OUTPUT_BASE/logs/ph2_${split}_xc.log
    run_stage --dataset PH2 --split $split --generate_concepts --data_path $DATA_PATH

    LOG_FILE=$OUTPUT_BASE/logs/ph2_${split}_cy.log
    run_stage --dataset PH2 --split $split --llm MMed --ckpt $CKPT_PATH --n_demos 0 --data_path $DATA_PATH

    echo "✓ PH2 split $split done"
done

# ============================================
# Derm7pt
# ============================================
echo "========== Derm7pt =========="

LOG_FILE=$OUTPUT_BASE/logs/derm7_xc.log
run_stage --dataset Derm7pt --generate_concepts --data_path $DATA_PATH

LOG_FILE=$OUTPUT_BASE/logs/derm7_cy.log
run_stage --dataset Derm7pt --llm MMed --ckpt $CKPT_PATH --n_demos 0 --data_path $DATA_PATH

# ============================================
# HAM10000
# ============================================
echo "========== HAM10000 =========="

LOG_FILE=$OUTPUT_BASE/logs/ham_xc.log
run_stage --dataset HAM10000 --generate_concepts --data_path $DATA_PATH

LOG_FILE=$OUTPUT_BASE/logs/ham_cy.log
run_stage --dataset HAM10000 --llm MMed --ckpt $CKPT_PATH --n_demos 0 --data_path $DATA_PATH

# ============================================
# DONE
# ============================================
echo "========================================="
echo "✅ PIPELINE FINISHED"
echo "Finished: $(date)"
echo "========================================="

echo "All outputs saved in: $OUTPUT_BASE"

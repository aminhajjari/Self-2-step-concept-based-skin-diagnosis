#!/bin/bash
#SBATCH --job-name=xcy_selfrefine
#SBATCH --account=def-arashmoh
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/gkianfar/scratch/Amin/concept/outputs/logs/xcy_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/concept/outputs/logs/xcy_%j.err

set -e  # stop on error

echo "========================================="
echo "🚀 FULL x→c→y PIPELINE WITH SELF-REFINE"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "========================================="

# ==============================
# LOAD MODULES
# ==============================
module load gcc python/3.11 cuda/12.6

# ==============================
# ACTIVATE ENV
# ==============================
source /home/gkianfar/scratch/Amin/conceptvenv/bin/activate
echo "✓ Environment: $VIRTUAL_ENV"

# ==============================
# PATHS
# ==============================
PROJECT_PATH="/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis"
DATA_PATH="$PROJECT_PATH/data"
OUTPUT_BASE="/home/gkianfar/scratch/Amin/concept/outputs"

cd $PROJECT_PATH || exit 1

# ==============================
# OUTPUT STRUCTURE
# ==============================
mkdir -p $OUTPUT_BASE/logs
mkdir -p $OUTPUT_BASE/results/concept_prediction
mkdir -p $OUTPUT_BASE/results/label_prediction

# Redirect results folder
[ -d results ] && rm -rf results
ln -s $OUTPUT_BASE/results results

echo "📁 DATA PATH: $DATA_PATH"
echo "📁 OUTPUT PATH: $OUTPUT_BASE"

# ==============================
# DEBUG DATA STRUCTURE
# ==============================
echo "🔎 Checking datasets..."
ls $DATA_PATH
ls $DATA_PATH/PH2
ls $DATA_PATH/Derm7pt
ls $DATA_PATH/HAM10000

# ==============================
# GPU INFO
# ==============================
echo ""
echo "🔍 GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

export CUDA_VISIBLE_DEVICES=$SLURM_GPUS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================
# PH2 (5 splits)
# ============================================
echo "========== PH2 =========="

for split in 0 1 2 3 4; do
    echo "---- PH2 Split $split ----"

    python run_x_to_c_to_y.py \
        --dataset PH2 \
        --split $split \
        --generate_concepts \
        --data_path $DATA_PATH \
        2>&1 | tee $OUTPUT_BASE/logs/ph2_${split}_xc.log

    python -c "import torch; torch.cuda.empty_cache()"
    sleep 2

    python run_x_to_c_to_y.py \
        --dataset PH2 \
        --split $split \
        --llm MMed \
        --ckpt Henrychur/MMed-Llama-3-8B \
        --n_demos 0 \
        --data_path $DATA_PATH \
        2>&1 | tee $OUTPUT_BASE/logs/ph2_${split}_cy.log

    python -c "import torch; torch.cuda.empty_cache()"
    echo "✓ PH2 split $split done"
done

# ============================================
# Derm7pt
# ============================================
echo "========== Derm7pt =========="

python run_x_to_c_to_y.py \
    --dataset Derm7pt \
    --generate_concepts \
    --data_path $DATA_PATH \
    2>&1 | tee $OUTPUT_BASE/logs/derm7_xc.log

python -c "import torch; torch.cuda.empty_cache()"
sleep 2

python run_x_to_c_to_y.py \
    --dataset Derm7pt \
    --llm MMed \
    --ckpt Henrychur/MMed-Llama-3-8B \
    --n_demos 0 \
    --data_path $DATA_PATH \
    2>&1 | tee $OUTPUT_BASE/logs/derm7_cy.log

# ============================================
# HAM10000
# ============================================
echo "========== HAM10000 =========="

python run_x_to_c_to_y.py \
    --dataset HAM10000 \
    --generate_concepts \
    --data_path $DATA_PATH \
    2>&1 | tee $OUTPUT_BASE/logs/ham_xc.log

python -c "import torch; torch.cuda.empty_cache()"
sleep 2

python run_x_to_c_to_y.py \
    --dataset HAM10000 \
    --llm MMed \
    --ckpt Henrychur/MMed-Llama-3-8B \
    --n_demos 0 \
    --data_path $DATA_PATH \
    2>&1 | tee $OUTPUT_BASE/logs/ham_cy.log

# ============================================
# DONE
# ============================================
echo "========================================="
echo "✅ PIPELINE FINISHED"
echo "Finished: $(date)"
echo "========================================="

echo ""
echo "📁 All outputs saved in:"
echo "$OUTPUT_BASE"

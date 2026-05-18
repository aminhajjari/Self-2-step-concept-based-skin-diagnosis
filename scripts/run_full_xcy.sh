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

module purge
module load gcc python/3.10 cuda/12.6 opencv/4.10.0
echo "✓ Modules loaded"

source /home/gkianfar/scratch/Amin/conceptvenv/bin/activate
echo "✓ Environment: $VIRTUAL_ENV"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TIMM_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/checkpoint/hf_cache
export HF_HUB_CACHE=$HF_HOME

echo "✓ Offline mode enforced"

PROJECT_PATH="/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis"
DATA_PATH="$PROJECT_PATH/data"
OUTPUT_BASE="/home/gkianfar/scratch/Amin/concept/outputs"
CKPT_PATH="$PROJECT_PATH/checkpoint/MMed-Llama-3-8B-EnIns"

cd $PROJECT_PATH || exit 1

mkdir -p $OUTPUT_BASE/logs
mkdir -p $OUTPUT_BASE/results/concept_prediction
mkdir -p $OUTPUT_BASE/results/label_prediction

[ -d results ] && rm -rf results
ln -s $OUTPUT_BASE/results results

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

echo "=== Checking Local Model Files ==="
BCLIP="$PROJECT_PATH/checkpoint/BiomedCLIP"
BBERT="$PROJECT_PATH/checkpoint/hf_cache/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/main"
ls -lh $BCLIP/open_clip_pytorch_model.bin 2>/dev/null && echo "✅ BiomedCLIP weights found" || echo "❌ BiomedCLIP weights MISSING"
ls -lh $BCLIP/open_clip_config.json       2>/dev/null && echo "✅ BiomedCLIP config found"  || echo "❌ BiomedCLIP config MISSING"
ls -lh $BBERT/open_clip_pytorch_model.bin 2>/dev/null && echo "✅ BiomedBERT weights found" || echo "❌ BiomedBERT weights MISSING"
echo ""

run_stage () {
    echo "Running: python run_x_to_c_to_y.py $@"
    python run_x_to_c_to_y.py "$@" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ ERROR: python run_x_to_c_to_y.py $@ failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 3
}

# ── PH2 (5 splits) ────────────────────────────────────────────────────────────
echo "========== PH2 =========="
for split in 0 1 2 3 4; do
    echo "---- PH2 Split $split ----"

    LOG_FILE=$OUTPUT_BASE/logs/ph2_${split}_xc.log
    run_stage --dataset PH2 --split $split \
              --model Explicd --concept_extractor Explicd \
              --generate_concepts --data_path $DATA_PATH

    LOG_FILE=$OUTPUT_BASE/logs/ph2_${split}_cy.log
    run_stage --dataset PH2 --split $split \
              --concept_extractor Explicd \
              --llm MMed --ckpt $CKPT_PATH \
              --n_demos 0 --data_path $DATA_PATH

    echo "✓ PH2 split $split done"
done

# ── Derm7pt & HAM10000 ────────────────────────────────────────────────────────
for dataset in Derm7pt HAM10000; do
    echo "========== $dataset =========="

    LOG_FILE=$OUTPUT_BASE/logs/${dataset}_xc.log
    run_stage --dataset $dataset \
              --model Explicd --concept_extractor Explicd \
              --generate_concepts --data_path $DATA_PATH

    LOG_FILE=$OUTPUT_BASE/logs/${dataset}_cy.log
    run_stage --dataset $dataset \
              --concept_extractor Explicd \
              --llm MMed --ckpt $CKPT_PATH \
              --n_demos 0 --data_path $DATA_PATH

    echo "✓ $dataset done"
done

# ── SUMMARY TABLES (printed at end of .out file) ──────────────────────────────
echo ""
echo "========================================="
echo "  PRINTING COMPARISON TABLES"
echo "========================================="

python evaluate_results.py

echo "========================================="
echo "✅ PIPELINE FINISHED"
echo "Finished: $(date)"
echo "========================================="
echo "All outputs saved in: $OUTPUT_BASE"

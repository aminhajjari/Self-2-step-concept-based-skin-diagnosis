#!/bin/bash
#SBATCH --job-name=xcy_3configs
#SBATCH --account=def-arashmoh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/gkianfar/scratch/Amin/concept/outputs/logs/xcy_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/concept/outputs/logs/xcy_%j.err

set -e

echo "========================================="
echo "  x→c→y PIPELINE — 6 CONFIGS COMPARISON"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "========================================="

# ── environment ────────────────────────────────────────────────────────────────
module purge
module load gcc python/3.10 cuda/12.6 opencv/4.10.0

source /home/gkianfar/scratch/Amin/conceptvenv/bin/activate

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

PROJECT_PATH="/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis"
DATA_PATH="$PROJECT_PATH/data"
OUTPUT_BASE="/home/gkianfar/scratch/Amin/concept/outputs"
MMED_CKPT="$PROJECT_PATH/checkpoint/MMed-Llama-3-8B-EnIns"
MISTRAL_CKPT="$PROJECT_PATH/checkpoint/Mistral-7B-Instruct"

cd $PROJECT_PATH || exit 1

mkdir -p $OUTPUT_BASE/logs
mkdir -p $OUTPUT_BASE/results/concept_prediction
mkdir -p $OUTPUT_BASE/results/label_prediction

[ -d results ] && rm -rf results
ln -s $OUTPUT_BASE/results results

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ── helper ─────────────────────────────────────────────────────────────────────
run_stage() {
    echo ">>> python run_x_to_c_to_y.py $@"
    python run_x_to_c_to_y.py "$@"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 2
}


# ══════════════════════════════════════════════════════════════════════════════
# PH2  (5-fold cross-validation)
# Generate concepts once per refiner type — reused across classifier configs
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════  PH2 — generating concepts (rule refiner)  ══════════"
for split in 0 1 2 3 4; do
    run_stage --dataset PH2 --split $split \
              --model Explicd --concept_extractor Explicd \
              --generate_concepts --data_path $DATA_PATH \
              --refiner rule
done

echo ""
echo "══════════  PH2 — generating concepts (mistral refiner)  ══════════"
for split in 0 1 2 3 4; do
    run_stage --dataset PH2 --split $split \
              --model Explicd --concept_extractor Explicd \
              --generate_concepts --data_path $DATA_PATH \
              --refiner mistral
done

echo ""
echo "══════════  PH2 — generating concepts (mmed refiner)  ══════════"
for split in 0 1 2 3 4; do
    run_stage --dataset PH2 --split $split \
              --model Explicd --concept_extractor Explicd \
              --generate_concepts --data_path $DATA_PATH \
              --refiner mmed
done

echo ""
echo "══════════  PH2 — c→y for all 6 configs  ══════════"

# Config A: rule refiner + MMed classifier
echo "--- PH2: Config A (Rule + MMed) ---"
for split in 0 1 2 3 4; do
    run_stage --dataset PH2 --split $split \
              --concept_extractor Explicd \
              --llm MMed --ckpt $MMED_CKPT \
              --n_demos 0 --refiner rule
done

# Config B: rule refiner + Mistral classifier
echo "--- PH2: Config B (Rule + Mistral) ---"
for split in 0 1 2 3 4; do
    run_stage --dataset PH2 --split $split \
              --concept_extractor Explicd \
              --llm Mistral --ckpt $MMED_CKPT \
              --classifier_ckpt $MISTRAL_CKPT \
              --n_demos 0 --refiner rule
done

# Config C: mistral refiner + Mistral classifier
echo "--- PH2: Config C (Mistral + Mistral) ---"
for split in 0 1 2 3 4; do
    run_stage --dataset PH2 --split $split \
              --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT \
              --n_demos 0 --refiner mistral
done

# Config D: mistral refiner + MMed classifier
echo "--- PH2: Config D (Mistral + MMed) ---"
for split in 0 1 2 3 4; do
    run_stage --dataset PH2 --split $split \
              --concept_extractor Explicd \
              --llm MMed --ckpt $MISTRAL_CKPT \
              --classifier_ckpt $MMED_CKPT \
              --n_demos 0 --refiner mistral
done

# Config E: mmed refiner + MMed classifier
echo "--- PH2: Config E (MMed + MMed) ---"
for split in 0 1 2 3 4; do
    run_stage --dataset PH2 --split $split \
              --concept_extractor Explicd \
              --llm MMed --ckpt $MMED_CKPT \
              --n_demos 0 --refiner mmed
done

# Config F: mmed refiner + Mistral classifier
echo "--- PH2: Config F (MMed + Mistral) ---"
for split in 0 1 2 3 4; do
    run_stage --dataset PH2 --split $split \
              --concept_extractor Explicd \
              --llm Mistral --ckpt $MMED_CKPT \
              --classifier_ckpt $MISTRAL_CKPT \
              --n_demos 0 --refiner mmed
done

# ══════════════════════════════════════════════════════════════════════════════
# Derm7pt and HAM10000
# ══════════════════════════════════════════════════════════════════════════════
for dataset in Derm7pt HAM10000; do
    echo ""
    echo "══════════  $dataset  ══════════"

    # --- concept generation (once per refiner) ---
    echo "--- $dataset: x→c (rule refiner) ---"
    run_stage --dataset $dataset \
              --model Explicd --concept_extractor Explicd \
              --generate_concepts --data_path $DATA_PATH \
              --refiner rule

    echo "--- $dataset: x→c (mistral refiner) ---"
    run_stage --dataset $dataset \
              --model Explicd --concept_extractor Explicd \
              --generate_concepts --data_path $DATA_PATH \
              --refiner mistral

    echo "--- $dataset: x→c (mmed refiner) ---"
    run_stage --dataset $dataset \
              --model Explicd --concept_extractor Explicd \
              --generate_concepts --data_path $DATA_PATH \
              --refiner mmed

    # --- Config A: rule + MMed ---
    echo "--- $dataset: Config A (Rule + MMed) ---"
    run_stage --dataset $dataset \
              --concept_extractor Explicd \
              --llm MMed --ckpt $MMED_CKPT \
              --n_demos 0 --refiner rule

    # --- Config B: rule + Mistral ---
    echo "--- $dataset: Config B (Rule + Mistral) ---"
    run_stage --dataset $dataset \
              --concept_extractor Explicd \
              --llm Mistral --ckpt $MMED_CKPT \
              --classifier_ckpt $MISTRAL_CKPT \
              --n_demos 0 --refiner rule

    # --- Config C: mistral + Mistral ---
    echo "--- $dataset: Config C (Mistral + Mistral) ---"
    run_stage --dataset $dataset \
              --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT \
              --n_demos 0 --refiner mistral

    # --- Config D: mistral + MMed ---
    echo "--- $dataset: Config D (Mistral + MMed) ---"
    run_stage --dataset $dataset \
              --concept_extractor Explicd \
              --llm MMed --ckpt $MISTRAL_CKPT \
              --classifier_ckpt $MMED_CKPT \
              --n_demos 0 --refiner mistral

    # --- Config E: mmed + MMed ---
    echo "--- $dataset: Config E (MMed + MMed) ---"
    run_stage --dataset $dataset \
              --concept_extractor Explicd \
              --llm MMed --ckpt $MMED_CKPT \
              --n_demos 0 --refiner mmed

    # --- Config F: mmed + Mistral ---
    echo "--- $dataset: Config F (MMed + Mistral) ---"
    run_stage --dataset $dataset \
              --concept_extractor Explicd \
              --llm Mistral --ckpt $MMED_CKPT \
              --classifier_ckpt $MISTRAL_CKPT \
              --n_demos 0 --refiner mmed

    echo "✓ $dataset done"
done
# ══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXPERIMENTS — all 6 configs x n_shots in {1, 2, 4, 8}
# Concepts already generated in zero-shot phase — reuse them (no --generate_concepts)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════  FEW-SHOT EXPERIMENTS — ALL 6 CONFIGS  ══════════"

for n_shots in 1 2 4 8; do
    echo ""
    echo "══════════  n_shots = $n_shots  ══════════"

    # ── PH2 ──────────────────────────────────────────────────────────────────
    echo "--- PH2: Config A (Rule + MMed) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm MMed --ckpt $MMED_CKPT \
                  --use_demos --n_demos $n_shots \
                  --refiner rule
    done

    echo "--- PH2: Config B (Rule + Mistral) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm Mistral --ckpt $MMED_CKPT \
                  --classifier_ckpt $MISTRAL_CKPT \
                  --use_demos --n_demos $n_shots \
                  --refiner rule
    done

    echo "--- PH2: Config C (Mistral + Mistral) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm Mistral --ckpt $MISTRAL_CKPT \
                  --use_demos --n_demos $n_shots \
                  --refiner mistral
    done

    echo "--- PH2: Config D (Mistral + MMed) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm MMed --ckpt $MISTRAL_CKPT \
                  --classifier_ckpt $MMED_CKPT \
                  --use_demos --n_demos $n_shots \
                  --refiner mistral
    done

    echo "--- PH2: Config E (MMed + MMed) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm MMed --ckpt $MMED_CKPT \
                  --use_demos --n_demos $n_shots \
                  --refiner mmed
    done

    echo "--- PH2: Config F (MMed + Mistral) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm Mistral --ckpt $MMED_CKPT \
                  --classifier_ckpt $MISTRAL_CKPT \
                  --use_demos --n_demos $n_shots \
                  --refiner mmed
    done

    # ── Derm7pt and HAM10000 ──────────────────────────────────────────────────
    for dataset in Derm7pt HAM10000; do
        echo "--- $dataset: Config A (Rule + MMed) ${n_shots}-shot ---"
        run_stage --dataset $dataset \
                  --concept_extractor Explicd \
                  --llm MMed --ckpt $MMED_CKPT \
                  --use_demos --n_demos $n_shots \
                  --refiner rule

        echo "--- $dataset: Config B (Rule + Mistral) ${n_shots}-shot ---"
        run_stage --dataset $dataset \
                  --concept_extractor Explicd \
                  --llm Mistral --ckpt $MMED_CKPT \
                  --classifier_ckpt $MISTRAL_CKPT \
                  --use_demos --n_demos $n_shots \
                  --
# ── final comparison table ─────────────────────────────────────────────────────
echo ""
echo "══════════  FINAL COMPARISON TABLES  ══════════"
python evaluate_results.py

echo ""
echo "========================================="
echo "  PIPELINE FINISHED"
echo "  Finished: $(date)"
echo "========================================="

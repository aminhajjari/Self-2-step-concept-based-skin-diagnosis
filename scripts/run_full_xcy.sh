#!/bin/bash
#SBATCH --job-name=xcy_fullexp
#SBATCH --account=def-arashmoh
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/gkianfar/scratch/Amin/concept/outputs/logs/xcy_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/concept/outputs/logs/xcy_%j.err

set -e

echo "========================================="
echo "  x→c→y PIPELINE — ZERO + FEW SHOT"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "========================================="

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
# STEP 1: GENERATE CONCEPTS (once per refiner — reused for all shot settings)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════  PH2 — generating concepts  ══════════"
for refiner in rule mistral mmed; do
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --model Explicd --concept_extractor Explicd \
                  --generate_concepts --predict_for_train_set \
                  --data_path $DATA_PATH \
                  --refiner $refiner
    done
done

echo ""
echo "══════════  Derm7pt + HAM10000 — generating concepts  ══════════"
for dataset in Derm7pt HAM10000; do
    for refiner in rule mistral mmed; do
        run_stage --dataset $dataset \
                  --model Explicd --concept_extractor Explicd \
                  --generate_concepts --predict_for_train_set \
                  --data_path $DATA_PATH \
                  --refiner $refiner
    done
done

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: ZERO-SHOT + FEW-SHOT CLASSIFICATION
# Loop over all n_shots (0 = zero-shot, 1/2/4/8 = few-shot)
# ══════════════════════════════════════════════════════════════════════════════
for n_shots in 0 1 2 4 8; do
    echo ""
    echo "══════════  n_shots = $n_shots — ALL 6 CONFIGS  ══════════"

    if [ $n_shots -eq 0 ]; then
        DEMOS_FLAG=""
    else
        DEMOS_FLAG="--use_demos"
    fi

    # ── PH2 ──────────────────────────────────────────────────────────────────
    echo "--- PH2: Config A (Rule + MMed) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm MMed --ckpt $MMED_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner rule
    done

    echo "--- PH2: Config B (Rule + Mistral) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm Mistral --ckpt $MMED_CKPT \
                  --classifier_ckpt $MISTRAL_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner rule
    done

    echo "--- PH2: Config C (Mistral + Mistral) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm Mistral --ckpt $MISTRAL_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner mistral
    done

    echo "--- PH2: Config D (Mistral + MMed) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm MMed --ckpt $MISTRAL_CKPT \
                  --classifier_ckpt $MMED_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner mistral
    done

    echo "--- PH2: Config E (MMed + MMed) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm MMed --ckpt $MMED_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner mmed
    done

    echo "--- PH2: Config F (MMed + Mistral) ${n_shots}-shot ---"
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --concept_extractor Explicd \
                  --llm Mistral --ckpt $MMED_CKPT \
                  --classifier_ckpt $MISTRAL_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner mmed
    done

    # ── Derm7pt and HAM10000 ─────────────────────────────────────────────────
    for dataset in Derm7pt HAM10000; do
        echo "--- $dataset: Config A (Rule + MMed) ${n_shots}-shot ---"
        run_stage --dataset $dataset \
                  --concept_extractor Explicd \
                  --llm MMed --ckpt $MMED_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner rule

        echo "--- $dataset: Config B (Rule + Mistral) ${n_shots}-shot ---"
        run_stage --dataset $dataset \
                  --concept_extractor Explicd \
                  --llm Mistral --ckpt $MMED_CKPT \
                  --classifier_ckpt $MISTRAL_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner rule

        echo "--- $dataset: Config C (Mistral + Mistral) ${n_shots}-shot ---"
        run_stage --dataset $dataset \
                  --concept_extractor Explicd \
                  --llm Mistral --ckpt $MISTRAL_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner mistral

        echo "--- $dataset: Config D (Mistral + MMed) ${n_shots}-shot ---"
        run_stage --dataset $dataset \
                  --concept_extractor Explicd \
                  --llm MMed --ckpt $MISTRAL_CKPT \
                  --classifier_ckpt $MMED_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner mistral

        echo "--- $dataset: Config E (MMed + MMed) ${n_shots}-shot ---"
        run_stage --dataset $dataset \
                  --concept_extractor Explicd \
                  --llm MMed --ckpt $MMED_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner mmed

        echo "--- $dataset: Config F (MMed + Mistral) ${n_shots}-shot ---"
        run_stage --dataset $dataset \
                  --concept_extractor Explicd \
                  --llm Mistral --ckpt $MMED_CKPT \
                  --classifier_ckpt $MISTRAL_CKPT \
                  $DEMOS_FLAG --n_demos $n_shots \
                  --refiner mmed

        echo "✓ $dataset ${n_shots}-shot done"
    done

done   # closes: for n_shots

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: PRINT FINAL TABLES
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════  ZERO-SHOT COMPARISON TABLE  ══════════"
python evaluate_results.py

echo ""
echo "══════════  ZERO + FEW-SHOT COMPARISON TABLE  ══════════"
python evaluate_fewshot_results.py

echo ""
echo "========================================="
echo "  PIPELINE FINISHED"
echo "  Finished: $(date)"
echo "========================================="

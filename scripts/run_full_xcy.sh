#!/bin/bash
#SBATCH --job-name=xcy_fullexp
#SBATCH --account=def-arashmoh_gpu
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
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

# Suppress Python warnings and debug prints from libraries
export PYTHONWARNINGS="ignore"

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

# ── run_stage: suppress all verbose output, only errors go to .err ──────────
# stdout (model loading, tqdm, DEBUG lines) → /dev/null
# stderr (real errors) → stays in .err via SLURM naturally
run_stage() {
    echo ">>> $(date '+%H:%M:%S') python run_x_to_c_to_y.py $@"
    # AFTER  — keep stderr (errors), only hide stdout (tqdm/debug noise)
    python run_x_to_c_to_y.py "$@" \
        1>/dev/null               # hide verbose stdout, keep real errors in .err
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: stage failed with exit code $EXIT_CODE — args: $@" >&2
        exit $EXIT_CODE
    fi
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 2
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: GENERATE CONCEPTS  (x → c)
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "══════════  STEP 1: Generating Concepts (x→c)  ══════════"

for refiner in rule mistral mmed; do
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split \
                  --model Explicd --concept_extractor Explicd \
                  --generate_concepts --predict_for_train_set \
                  --data_path $DATA_PATH \
                  --refiner $refiner
    done
    echo "  PH2 refiner=$refiner done"
done

for dataset in Derm7pt HAM10000; do
    for refiner in rule mistral mmed; do
        run_stage --dataset $dataset \
                  --model Explicd --concept_extractor Explicd \
                  --generate_concepts --predict_for_train_set \
                  --data_path $DATA_PATH \
                  --refiner $refiner
        echo "  $dataset refiner=$refiner done"
    done
done

echo "✓ STEP 1 complete"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: ZERO-SHOT + FEW-SHOT CLASSIFICATION  (c → y)  — ALL 6 CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "══════════  STEP 2: Classification (c→y) — All 6 Configs  ══════════"

for n_shots in 0 1 2 4 8; do

    echo ""
    echo "  ── n_shots=$n_shots ──"

    if [ $n_shots -eq 0 ]; then
        DEMOS_FLAG=""
        HAM_DEMOS=0
    elif [ $n_shots -gt 2 ]; then
        DEMOS_FLAG="--use_demos"
        HAM_DEMOS=2
    else
        DEMOS_FLAG="--use_demos"
        HAM_DEMOS=$n_shots
    fi

    # ── PH2 ──────────────────────────────────────────────────────────────
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner rule
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner rule
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner mistral
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm MMed    --ckpt $MMED_CKPT    --classifier_ckpt $MMED_CKPT \
                                                      $DEMOS_FLAG --n_demos $n_shots --refiner mistral
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner mmed
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner mmed
    done
    echo "  PH2 ${n_shots}-shot done"

    # ── Derm7pt ──────────────────────────────────────────────────────────
    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner rule
    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner rule
    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner mistral
    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner mistral
    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner mmed
    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner mmed
    echo "  Derm7pt ${n_shots}-shot done"

    # ── HAM10000 (capped at 2-shot) ───────────────────────────────────
    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $HAM_DEMOS --refiner rule
    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $HAM_DEMOS --refiner rule
    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $HAM_DEMOS --refiner mistral
    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $HAM_DEMOS --refiner mistral
    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $HAM_DEMOS --refiner mmed
    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $HAM_DEMOS --refiner mmed
    echo "  HAM10000 ${HAM_DEMOS}-shot done"

done

echo "✓ STEP 2 complete"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2b: ABLATION  (Rule-refined concepts → each LLM)
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "══════════  STEP 2b: Ablation — Rule concepts → each LLM  ══════════"

RULE_D7="results/concept_prediction/Derm7pt_dermatology_reports_generated_by_Explicd_refiner_rule_raw_values_False.csv"
RULE_HAM="results/concept_prediction/HAM10000_dermatology_reports_generated_by_Explicd_refiner_rule_raw_values_False.csv"

for n_shots in 0 1 2; do
    if [ $n_shots -eq 0 ]; then DEMOS_FLAG=""; else DEMOS_FLAG="--use_demos"; fi

    run_stage --dataset Derm7pt  --concept_extractor Explicd --report_path $RULE_D7  \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner rule
    run_stage --dataset Derm7pt  --concept_extractor Explicd --report_path $RULE_D7  \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner rule
    run_stage --dataset HAM10000 --concept_extractor Explicd --report_path $RULE_HAM \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner rule
    run_stage --dataset HAM10000 --concept_extractor Explicd --report_path $RULE_HAM \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner rule
    echo "  Ablation ${n_shots}-shot done"
done

echo "✓ STEP 2b complete"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2c: RANDOM DEMO BASELINE — ALL 6 CONFIGS (1-shot and 2-shot only)
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "══════════  STEP 2c: Random Demo Baseline  ══════════"

for n_shots in 1 2; do
    DEMOS_FLAG="--use_demos"

    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner rule    --random_demos
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner rule    --random_demos
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner mistral --random_demos
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner mistral --random_demos
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner mmed    --random_demos
        run_stage --dataset PH2 --split $split --concept_extractor Explicd \
                  --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner mmed    --random_demos
    done

    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner rule    --random_demos
    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner rule    --random_demos
    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner mistral --random_demos
    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner mistral --random_demos
    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner mmed    --random_demos
    run_stage --dataset Derm7pt --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner mmed    --random_demos

    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner rule    --random_demos
    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner rule    --random_demos
    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner mistral --random_demos
    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner mistral --random_demos
    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm MMed    --ckpt $MMED_CKPT    $DEMOS_FLAG --n_demos $n_shots --refiner mmed    --random_demos
    run_stage --dataset HAM10000 --concept_extractor Explicd \
              --llm Mistral --ckpt $MISTRAL_CKPT $DEMOS_FLAG --n_demos $n_shots --refiner mmed    --random_demos

    echo "  Random baseline ${n_shots}-shot done"
done

echo "✓ STEP 2c complete"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: PRINT FINAL TABLES  (these go to .out — clean, no noise)
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  STEP 3: RESULTS TABLES"
echo "════════════════════════════════════════════════════════════"

echo ""
echo "══════════  ZERO-SHOT COMPARISON TABLE  ══════════"
python evaluate_results.py

echo ""
echo "══════════  ZERO + FEW-SHOT COMPARISON TABLE  ══════════"
python evaluate_fewshot_results.py

echo ""
echo "══════════  REFINEMENT ANALYSIS TABLE  ══════════"
python analyze_refinement.py

echo ""
echo "══════════  STATISTICAL SIGNIFICANCE TESTS  ══════════"
python mcnemar_test.py

echo ""
echo "========================================="
echo "  PIPELINE FINISHED"
echo "  Finished: $(date)"
echo "========================================="

#!/bin/bash
#SBATCH --job-name=xcy_selfrefine
#SBATCH --account=def-arashmoh
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1              # Request H100 80GB GPU
#SBATCH --cpus-per-task=8              # Increased for faster processing
#SBATCH --mem=64G
#SBATCH --output=logs/xcy_pipeline_%j.out
#SBATCH --error=logs/xcy_pipeline_%j.err

echo "========================================="
echo "🚀 FULL x→c→y PIPELINE WITH SELF-REFINE"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date)"
echo "========================================="

# Load modules
module load gcc python/3.11 cuda/12.6 opencv scipy-stack

# Activate environment
source /project/def-arashmoh/shahab33/Medsam/self/bin/activate
echo "✓ Environment activated: $VIRTUAL_ENV"

# Go to project directory
cd /project/def-arashmoh/shahab33/Medsam/selff-ref || exit 1

# Create directories
mkdir -p logs results/concept_prediction results/label_prediction

# Set paths and CUDA environment
export CUDA_VISIBLE_DEVICES=0
export DATA_PATH="/home/gkianfar/scratch/Amin/maincode/Self-2-step-concept-based-skin-diagnosis/data"
export PYTORCH_ALLOC_CONF=expandable_segments:True  # Better memory management (new name)

# Check GPU
echo ""
echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

echo ""
echo "📊 Processing all datasets with self-refine..."
echo ""

# ============================================
# DATASET 1: PH2 (5 splits)
# ============================================
echo "========================================="
echo "[1/3] PH2 Dataset - 5 Splits"
echo "========================================="

for split in 0 1 2 3 4; do
    echo ""
    echo "--- PH2 Split $split ---"
    
    # Step 1: x→c (generate refined concepts)
    echo "  [x→c] Generating refined concepts..."
    python run_x_to_c_to_y.py \
        --dataset PH2 \
        --split $split \
        --generate_concepts \
        --data_path $DATA_PATH \
        2>&1 | tee -a logs/ph2_split_${split}_xc.log
    
    # Check if concept generation succeeded
    if [ $? -ne 0 ]; then
        echo "  ❌ ERROR: Concept generation failed for PH2 split $split"
        continue
    fi
    
    # Clear GPU memory between steps
    echo "  🧹 Clearing GPU memory..."
    python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
    sleep 2
    
    # Step 2: c→y (concepts to diagnosis)
    echo "  [c→y] Predicting diagnosis..."
    python run_x_to_c_to_y.py \
        --dataset PH2 \
        --split $split \
        --llm MMed \
        --ckpt Henrychur/MMed-Llama-3-8B \
        --n_demos 0 \
        --data_path $DATA_PATH \
        2>&1 | tee -a logs/ph2_split_${split}_cy.log
    
    if [ $? -ne 0 ]; then
        echo "  ❌ ERROR: Diagnosis prediction failed for PH2 split $split"
        continue
    fi
    
    # Clear GPU memory after each split
    echo "  🧹 Clearing GPU memory..."
    python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
    
    echo "  ✓ Split $split completed at $(date)"
    echo ""
done

echo ""
echo "✓ PH2 completed!"
echo ""

# ============================================
# DATASET 2: Derm7pt
# ============================================
echo "========================================="
echo "[2/3] Derm7pt Dataset"
echo "========================================="

# Step 1: x→c
echo "  [x→c] Generating refined concepts..."
python run_x_to_c_to_y.py \
    --dataset Derm7pt \
    --generate_concepts \
    --data_path $DATA_PATH \
    2>&1 | tee logs/derm7pt_xc.log

if [ $? -eq 0 ]; then
    # Clear GPU memory
    echo "  🧹 Clearing GPU memory..."
    python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
    sleep 2
    
    # Step 2: c→y
    echo "  [c→y] Predicting diagnosis..."
    python run_x_to_c_to_y.py \
        --dataset Derm7pt \
        --llm MMed \
        --ckpt Henrychur/MMed-Llama-3-8B \
        --n_demos 0 \
        --data_path $DATA_PATH \
        2>&1 | tee logs/derm7pt_cy.log
    
    echo ""
    echo "✓ Derm7pt completed!"
else
    echo "  ❌ ERROR: Concept generation failed for Derm7pt"
fi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
echo ""

# ============================================
# DATASET 3: HAM10000
# ============================================
echo "========================================="
echo "[3/3] HAM10000 Dataset"
echo "========================================="

# Step 1: x→c
echo "  [x→c] Generating refined concepts..."
python run_x_to_c_to_y.py \
    --dataset HAM10000 \
    --generate_concepts \
    --data_path $DATA_PATH \
    2>&1 | tee logs/ham10000_xc.log

if [ $? -eq 0 ]; then
    # Clear GPU memory
    echo "  🧹 Clearing GPU memory..."
    python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"
    sleep 2
    
    # Step 2: c→y
    echo "  [c→y] Predicting diagnosis..."
    python run_x_to_c_to_y.py \
        --dataset HAM10000 \
        --llm MMed \
        --ckpt Henrychur/MMed-Llama-3-8B \
        --n_demos 0 \
        --data_path $DATA_PATH \
        2>&1 | tee logs/ham10000_cy.log
    
    echo ""
    echo "✓ HAM10000 completed!"
else
    echo "  ❌ ERROR: Concept generation failed for HAM10000"
fi

echo ""

# ============================================
# SUMMARY
# ============================================
echo "========================================="
echo "✅ COMPLETE PIPELINE FINISHED"
echo "========================================="
echo "Completed: $(date)"
echo ""
echo "📁 Results saved in:"
echo "  Concepts: results/concept_prediction/"
echo "  Diagnosis: results/label_prediction/"
echo ""
echo "📊 To view results:"
echo "  cd results/label_prediction"
echo "  ls -lh"
echo ""
echo "📊 Result files:"
ls -lh results/label_prediction/ 2>/dev/null || echo "  (No files yet)"
echo ""
echo "📈 Next: Calculate metrics with:"
echo "  python calculate_metrics.py --model=MMed --task=PH2_eval ..."
echo "========================================="

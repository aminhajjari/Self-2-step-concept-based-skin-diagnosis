#!/bin/bash
#SBATCH --job-name=multi_expert
#SBATCH --account=def-arashmoh_gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/gkianfar/scratch/Amin/concept/outputs/logs/multi_expert_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/concept/outputs/logs/multi_expert_%j.err

set -e
module purge
module load gcc python/3.10 cuda/12.6 opencv/4.10.0
source /home/gkianfar/scratch/Amin/conceptvenv/bin/activate

PROJECT_PATH="/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis"
OUTPUT_BASE="/home/gkianfar/scratch/Amin/concept/outputs"
cd $PROJECT_PATH

# make sure results/ points at the shared output dir (same symlink pattern as run_full_xcy.sh)
mkdir -p $OUTPUT_BASE/results/multi_expert $OUTPUT_BASE/results/tables
[ -e results ] || ln -s $OUTPUT_BASE/results results

echo "=========================================="
echo "  MULTI-EXPERT ENSEMBLE — MMed + Mistral + MedGemma"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "=========================================="

# One run per (n_demos, retrieval) setting actually present in results/label_prediction.
# rices sweep: 0,1,2,4,8-shot ; random-demo baseline sweep: 1,2-shot
python multi_expert_classifier.py \
    --experts MMed Mistral MedGemma \
    --sweep 0:rices 1:rices 2:rices 4:rices 8:rices 1:random 2:random \
    --tie melanoma \
    --prob_col p_melanoma

echo ""
echo "=========================================="
echo "  DONE — tables in results/tables/"
echo "  Finished: $(date)"
echo "=========================================="
ls -la results/tables/

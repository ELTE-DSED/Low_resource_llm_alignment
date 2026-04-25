#!/bin/bash
# =============================================================================
#  SLURM BATCH SCRIPT — Curriculum RL Training
#  Usage:
#    sbatch submit_curriculum.sh          # fresh run
#    sbatch submit_curriculum.sh          # resume — state is read automatically
# =============================================================================

# ── Resource allocation ───────────────────────────────────────────────────────
#SBATCH --job-name=curriculum_rl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ── Output & error logs ───────────────────────────────────────────────────────
#SBATCH --output=logs/slurm_%j.out       # %j = job ID
#SBATCH --error=logs/slurm_%j.err
#SBATCH --open-mode=append               # don't overwrite logs on requeue

# ── Email notifications ───────────────────────────────────────────────────────
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com   # ← set your email here

# ── Signal script 5 minutes before wall-time so it saves state cleanly ───────
#SBATCH --signal=USR1@300                # SIGUSR1 sent 300s before timeout

# ── Auto-requeue: job re-submits itself after each 12h block ─────────────────
#SBATCH --requeue

# =============================================================================
set -euo pipefail

# ── EDIT THESE ────────────────────────────────────────────────────────────────
CONDA_ENV="distillation_project"
WORK_DIR="/singularity/100-gpu01/arafat_data/distillation_project/komondoro_test/Komondor codebase"

# ── Helper ────────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "======================================================"
log " Job ID     : $SLURM_JOB_ID"
log " Job name   : $SLURM_JOB_NAME"
log " Node       : $(hostname)"
log " GPU        : ${SLURM_JOB_GPUS:-0}"
log " Work dir   : $WORK_DIR"
log " Start time : $(date)"
log "======================================================"

# ── Activate conda environment ────────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
log "Activated conda env: $CONDA_ENV  (Python: $(python --version))"

# ── Environment variables ─────────────────────────────────────────────────────
export HF_TOKEN="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"
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS:-0}"
export TOKENIZERS_PARALLELISM=false      # suppress HuggingFace fork warning
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$WORK_DIR"
mkdir -p logs
log "Working directory: $(pwd)"

log "GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

log "Launching Experiments_run.py ..."
python Experiments_run.py

EXIT_CODE=$?
log "Experiments_run.py exited with code $EXIT_CODE"

log "Disk usage at exit:"
df -h .

if [ $EXIT_CODE -eq 0 ]; then
    log "✅ Curriculum training completed successfully."
else
    log "❌ Job failed with exit code $EXIT_CODE — check logs/ for details."
fi

exit $EXIT_CODE
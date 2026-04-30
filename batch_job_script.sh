

#SBATCH --job-name=curriculum_rl
#SBATCH --partition=test ## Should have a GPU                 
#SBATCH --account= user_name            
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@gmail.com   

set -euo pipefail

# ── STORAGE LAYOUT ────────────────────────────────────────────────────────────
#
#   ~/dirctory/         — code, checkpoints, final outputs (persistent)
#   ~/directory_scratch/  — active job data, logs, HF cache  (fast, temporary)
#
#  Everything written during training goes to scratch.
#  Copy important outputs to /project after the job finishes.


PROJECT_DIR="$HOME/dir_2026"
SCRATCH_DIR="$HOME/dir_scratch"
WORK_DIR="$PROJECT_DIR/curriculum_rl"

VENV_PATH="$PROJECT_DIR/.curriculum_rl"

HF_CACHE="$SCRATCH_DIR/.cache/huggingface"

STATE_DIR="$SCRATCH_DIR/curriculum_state"

# Store token in ~/.hf_token  (chmod 600 ~/.hf_token)
HF_TOKEN_VALUE=$(cat ~/.hf_token 2>/dev/null || echo "YOUR_HF_TOKEN")

# ── Helper ────────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Create required directories ───────────────────────────────────────────────
mkdir -p "$SCRATCH_DIR/logs"
mkdir -p "$HF_CACHE"
mkdir -p "$STATE_DIR"

log "======================================================"
log " Job ID     : $SLURM_JOB_ID"
log " Job name   : $SLURM_JOB_NAME"
log " Node       : $(hostname)"
log " GPU        : ${SLURM_JOB_GPUS:-0}"
log " Work dir   : $WORK_DIR"
log " Scratch    : $SCRATCH_DIR"
log " Start time : $(date)"
log "======================================================"

log "Storage quotas:"
squota 2>/dev/null || df -h "$PROJECT_DIR" "$SCRATCH_DIR"

# ── Load HPC modules ──────────────────────────────────────────────────────────
module purge
module load cray-python/3.10            
module load cuda/12.4
log "Loaded modules:"
module list 2>&1

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    log "ERROR: venv not found at $VENV_PATH — did you create it on a login node?"
    exit 1
fi
source "$VENV_PATH/bin/activate"
log "Activated venv: $VENV_PATH  (Python: $(python --version))"

# ── Package sanity check ──────────────────────────────────────────────────────
python - <<'EOF'
import torch, transformers
print(f"torch         : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version  : {torch.version.cuda}")
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
print(f"transformers  : {transformers.__version__}")
EOF

# ── Environment variables ─────────────────────────────────────────────────────
export HF_TOKEN="....." ## Your HF Token
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE/transformers"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS:-0}"
export BNB_CUDA_VERSION=124
export TOKENIZERS_PARALLELISM=false



ln -sf "$STATE_DIR/curriculum_state.json" "$WORK_DIR/curriculum_state.json"

# ── Change to working directory ───────────────────────────────────────────────
cd "$WORK_DIR"
log "Working directory: $(pwd)"

# ── GPU health check ──────────────────────────────────────────────────────────
log "GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# ── Run the curriculum pipeline ───────────────────────────────────────────────
log "Launching Experiments_run.py ..."
python Experiments_run.py

EXIT_CODE=$?
log "Experiments_run.py exited with code $EXIT_CODE"

# ── Quota report at exit ──────────────────────────────────────────────────────
log "Storage quotas at exit:"
squota 2>/dev/null || df -h "$PROJECT_DIR" "$SCRATCH_DIR"

# ── Copy key outputs from scratch → project on success ───────────────────────
if [ $EXIT_CODE -eq 0 ]; then
    log "Syncing checkpoints and logs from scratch to project storage ..."
    rsync -av --progress \
        "$SCRATCH_DIR/curriculum_state/" \
        "$PROJECT_DIR/curriculum_state_backup/"
    log "✅ Curriculum training completed successfully."
else
    log "❌ Job failed with exit code $EXIT_CODE — check $SCRATCH_DIR/logs/ for details."
fi

exit $EXIT_CODE

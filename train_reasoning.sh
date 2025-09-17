#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48                              ## Give the single process lots of CPU

#SBATCH --mem=96G                                       ## RAM per node
#SBATCH --time=12:00:00                                  ## Walltime
#SBATCH --gres=gpu:8                                    ## Number of GPUs
#SBATCH --exclude=neu[301,306]                          ## Exclude some nodes
#SBATCH --job-name=train_orz                            ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Error File
#SBATCH --error=slurm_outputs/%x/err_log_%x_%j.err      ## Output File
#SBATCH --mail-type=BEGIN,END,FAIL                      ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=jeremy.bao@princeton.edu


cd /n/fs/jborz/projects/Open-Reasoner-Zero
module load cudatoolkit/12.1
source .venv/bin/activate

# Force Ray to start locally, not connect to existing cluster
export RAY_ADDRESS=""
export RAY_DISABLE_IMPORT_WARNING="1"

# Prefer local (non-NFS) caches for speed/stability
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_cache"
export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR:-/tmp}/torch_extensions"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

# CUDA memory allocator: reduce fragmentation for long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SLURM-specific threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# launch under srun so Slurm tracks GPU usage
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --gpu-bind=closest \
  python -m playground.orz_0p5b_ppo

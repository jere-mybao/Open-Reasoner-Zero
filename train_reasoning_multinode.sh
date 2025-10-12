#!/bin/bash

set -euo pipefail

# If not running under Slurm, auto-submit to avoid login-node execution.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[INFO] Not inside a Slurm job. Submitting via sbatch to avoid running on the login node..."
  exec sbatch "$0"
fi

#SBATCH --nodes=2                                       ## Node count
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=48                              ## CPU per task
#SBATCH --mem=96G                                       ## RAM per node
#SBATCH --time=12:00:00                                 ## Walltime
#SBATCH --gres=gpu:8                                    ## GPUs per node
#SBATCH --exclude=neu[301,306]                          ## Exclude some nodes
#SBATCH --job-name=train_orz_multi                      ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Stdout
#SBATCH --error=slurm_outputs/%x/err_log_%x_%j.err      ## Stderr
#SBATCH --mail-type=BEGIN,END,FAIL                      ## Mail events
#SBATCH --mail-user=jeremy.bao@princeton.edu

cd /n/fs/jborz/projects/Open-Reasoner-Zero
module load cudatoolkit/12.1
source .venv/bin/activate

# Per-node caches (local storage)
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_cache"
export TORCH_EXTENSIONS_DIR="${SLURM_TMPDIR:-/tmp}/torch_extensions"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}

mkdir -p slurm_outputs/train_orz_multi

# Determine head node hostname/IP and free port
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HEAD_PORT=6379
# Resolve an IP for the head node that other nodes can reach
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "hostname -I | awk '{print \$1}'" | tail -n 1)

# Start Ray head on rank 0, and Ray worker on rank > 0
# Use srun to launch per-node commands
srun --export=ALL --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  ray stop >/dev/null 2>&1 || true
  ray start --head --port=$HEAD_PORT --dashboard-host=0.0.0.0 --num-cpus=${SLURM_CPUS_ON_NODE:-48} --disable-usage-stats
" &

# Start workers on the remaining nodes
OTHER_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2)
for node in $OTHER_NODES; do
  srun --export=ALL --nodes=1 --ntasks=1 -w "$node" bash -lc "
    ray stop >/dev/null 2>&1 || true
    ray start --address='$HEAD_IP:$HEAD_PORT' --num-cpus=${SLURM_CPUS_ON_NODE:-48} --disable-usage-stats
  " &
done

wait

# Run the training on the head node, connecting to the cluster
srun --export=ALL --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "
  export RAY_ADDRESS='$HEAD_IP:$HEAD_PORT'
  export RAY_DISABLE_IMPORT_WARNING=1
  python -m playground.orz_0p5b_ppo_multinode
"

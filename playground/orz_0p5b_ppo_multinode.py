"""
Qwen2.5-0.5B + PPO (2-node Ray cluster)

Assumptions:
- 2 nodes, each with multiple GPUs (e.g., 8 L40s per node).
- 10GbE (no Infiniband) → prefer TP within a node; scale vLLM via multiple engines.

Run via SLURM script that starts Ray head/workers, then:
  python -m playground.orz_0p5b_ppo_multinode
"""

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

# Project root on import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from omegaconf.listconfig import ListConfig

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExpConfig
from playground.orz_7b_ppo import PPOExp

file_name = f"{os.path.splitext(os.path.basename(__file__))[0]}"

executor = ThreadPoolExecutor(max_workers=64)


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # Use 2 nodes; Ray cluster is started by SLURM script
    total_num_nodes: int = 2

    # Non-colocated layout for stability across nodes
    ref_num_nodes: int = total_num_nodes
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = total_num_nodes
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = total_num_nodes
    critic_num_gpus_per_node: int = 1
    colocate_all: bool = False
    colocate_critic_reward: bool = False
    colocate_actor_ref: bool = False

    # vLLM engines: spread across nodes (Ray scheduler will place them)
    # Pick a safe default that doesn’t consume all GPUs; tune upward if headroom exists.
    vllm_num_engines: int = 6
    vllm_tensor_parallel_size: int = 1  # keep TP within a node on 10GbE

    adam_offload: bool = False
    zero_stage: int = 3

    # Paths
    pretrain: Optional[str] = "Qwen/Qwen2.5-0.5B"
    reward_pretrain: Optional[str] = None
    save_interval: int = 50
    ckpt_path: str = f"orz_ckpt/{file_name}"
    save_path: str = f"orz_ckpt/{file_name}"
    tensorboard_log_dir: str = f"orz_logs/{file_name}"

    # Datasets
    prompt_data: ListConfig = ListConfig([
        "data/orz_math_57k_collected.json",
    ])
    eval_prompt_data: ListConfig = ListConfig([
        "data/eval_data/math500.json",
        "data/eval_data/aime2024.json",
        "data/eval_data/gpqa_diamond.json",
    ])
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # PPO/training knobs (can be tuned based on memory/throughput)
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 50
    prompt_max_len: int = 2048
    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = True

    num_episodes: int = 20
    rollout_batch_size: int = 64  # start moderate across nodes
    n_samples_per_prompt: int = 16
    micro_rollout_batch_size: int = 64

    policy_update_steps: int = 1
    critic_update_steps: int = 12
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    kl_loss_coef: float = 0.0
    use_kl_loss: bool = True
    use_kl_estimator_k3: bool = True

    enable_eval: bool = True
    eval_interval: int = 10

    # Generation settings: consider reducing max len if OOM/slow
    generate_max_len: int = 4096
    max_len: int = 4096
    packing_max_len: int = generate_max_len + prompt_max_len
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>"])

    use_grpo: bool = False
    gpu_memory_utilization: float = 0.8
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    gamma: float = 1.0
    lambd: float = 1.0


if __name__ == "__main__":
    exp = PPOExp().set_cfg(PPOExpConfig())
    logger.info(exp.get_cfg_as_str(exp.cfg))
    os.makedirs(exp.cfg.save_path, exist_ok=True)
    os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    asyncio.run(exp.run())

#!/bin/bash
#SBATCH --job-name=rag-kc-llama3-8b-finetune
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/rag-kc-llama3-8b-finetune.%j.out
#SBATCH --error=logs/rag-kc-llama3-8b-finetune.%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40_ind_bias
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --time=1:00:00

# Activate env
source /fs01/projects/aieng/public/vector-eval/vectorlm_env/bin/activate

export NCCL_IB_DISABLE=1  # Our cluster does not have InfiniBand. We need to disable usage using this flag.
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN

# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Uncomment these flags for debugging communication
# export TORCH_CPP_LOG_LEVEL=INFO
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=0

torchrun --nnodes=1 --nproc-per-node=${SLURM_GPUS_ON_NODE} finetune.py --yaml_path configs/config.yaml

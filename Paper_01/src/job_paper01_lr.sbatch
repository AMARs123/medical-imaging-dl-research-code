#!/bin/bash -l
#SBATCH --job-name=p01_lr
#SBATCH --qos=faculty
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/Journal_Papers_Research/Paper_01/experiments/slurm_%j.out
# Learning-rate check of Paper 01 (lr_registry.csv, EXP771-EXP850), 17 Oct 2026, run before the disk-quota raise:
# P01_WEIGHTS_SCRATCH keeps the model weights of these runs in node-local scratch outside /home during training and removes
# them afterwards (see train.py); logs, metrics, predictions and evidence maps are written to the experiment folders as usual.
# The check only compares validation-fold scores, which do not need the stored weights.
R=$HOME/Journal_Papers_Research
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export P01_WEIGHTS_SCRATCH=/tmp/${USER}_p01_lr_weights
mkdir -p $P01_WEIGHTS_SCRATCH
PY=$R/shared_resources/envs/jpr/bin/python
cd $R/Paper_01/src
REGISTRY=$R/Paper_01/experiments/lr_registry.csv $PY run_batch.py $R/Paper_01 30 forward
echo P01_LR_DONE

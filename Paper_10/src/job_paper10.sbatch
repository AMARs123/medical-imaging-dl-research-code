#!/bin/bash -l
#SBATCH --job-name=p10_runs
#SBATCH --qos=faculty
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/Journal_Papers_Research/Paper_10/experiments/slurm_%j.out

# Runs all planned Paper 10 experiments without results (finished ones are skipped; up to three such jobs can
# share the work, see run_batch.py). ORDER=reverse makes a job start from the end of the registry.
R=$HOME/Journal_Papers_Research
PY=$R/shared_resources/envs/jpr/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export P10_DATA=/var/tmp/${USER}_data/camelyon17_np   # arrays from prepare_hf.py (outside the /home quota)
export WEIGHTS_DIR=${WEIGHTS_DIR:-/var/tmp/${USER}_weights/Paper_10}   # weights kept outside /home; location recorded per run
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
$PY $R/Paper_10/src/run_batch.py $R/Paper_10 70 ${ORDER:-forward}

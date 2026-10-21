#!/bin/bash -l
#SBATCH --job-name=p02_runs
#SBATCH --qos=faculty
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/Journal_Papers_Research/Paper_02/experiments/slurm_%j.out

# Runs all planned Paper 02 experiments without results (finished ones are skipped; up to three such jobs can
# share the work, see run_batch.py). ORDER=reverse makes a job start from the end of the registry.
R=$HOME/Journal_Papers_Research
PY=$R/shared_resources/envs/jpr/bin/python
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
$PY $R/Paper_02/src/run_batch.py $R/Paper_02 70 ${ORDER:-forward}

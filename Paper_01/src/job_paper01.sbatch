#!/bin/bash -l
#SBATCH --job-name=p01_runs
#SBATCH --qos=faculty
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/Journal_Papers_Research/Paper_01/experiments/slurm_%j.out

# Runs all planned Paper 01 experiments that have no results yet (finished ones are skipped, so this job can be
# resubmitted). Two such jobs may run together: experiments are claimed one at a time (see run_batch.py).
# Peak GPU memory of any model so far: 2.7 GB, so 3 shards (about 6 GB) are enough.
# ORDER=reverse makes a second job work from the end of the registry.
R=$HOME/Journal_Papers_Research
PY=$R/shared_resources/envs/jpr/bin/python
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
$PY $R/Paper_01/src/run_batch.py $R/Paper_01 70 ${ORDER:-forward}

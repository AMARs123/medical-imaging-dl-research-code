#!/bin/bash -l
#SBATCH --job-name=p08_runs
#SBATCH --qos=faculty
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/imaging_research/waste-sorting-edge/experiments/slurm_%j.out

# Runs all planned skin-lesion-model-ranking experiments without results (finished ones are skipped; up to three such jobs can
# share the work, see run_batch.py). ORDER=reverse makes a job start from the end of the registry.
R=$HOME/imaging_research
PY=$R/shared_resources/envs/jpr/bin/python
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
$PY $R/waste-sorting-edge/src/run_batch.py $R/waste-sorting-edge 70 ${ORDER:-forward}
$PY $R/waste-sorting-edge/src/efficiency.py $R/waste-sorting-edge   # CPU measurements on the cores allocated to this job

#!/bin/bash -l
#SBATCH --job-name=p05_runs
#SBATCH --qos=faculty
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/imaging_research/diabetic-retinopathy-ordinal-grading/experiments/slurm_%j.out

# Runs all planned skin-lesion-model-ranking experiments without results (finished ones are skipped; up to three such jobs can
# share the work, see run_batch.py). ORDER=reverse makes a job start from the end of the registry.
R=$HOME/imaging_research
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$R/shared_resources/envs/jpr/bin/python
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
$PY $R/diabetic-retinopathy-ordinal-grading/src/run_batch.py $R/diabetic-retinopathy-ordinal-grading 70 ${ORDER:-forward}

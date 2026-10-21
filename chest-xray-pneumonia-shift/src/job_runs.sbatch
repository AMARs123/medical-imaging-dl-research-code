#!/bin/bash -l
#SBATCH --job-name=p04_runs
#SBATCH --qos=faculty
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/imaging_research/chest-xray-pneumonia-shift/experiments/slurm_%j.out

# Runs all planned skin-lesion-model-ranking experiments without results (finished ones are skipped; up to three such jobs can
# share the work, see run_batch.py). ORDER=reverse makes a job start from the end of the registry.
R=$HOME/imaging_research
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$R/shared_resources/envs/jpr/bin/python
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
$PY $R/chest-xray-pneumonia-shift/src/run_batch.py $R/chest-xray-pneumonia-shift 70 ${ORDER:-forward}
# after all training runs: AdaBN and corruption-robustness evaluations of the stored models (no retraining)
$PY $R/chest-xray-pneumonia-shift/src/evaluate_extra.py $R/chest-xray-pneumonia-shift

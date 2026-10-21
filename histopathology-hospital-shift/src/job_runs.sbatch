#!/bin/bash -l
#SBATCH --job-name=p10_runs
#SBATCH --qos=faculty
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/imaging_research/histopathology-hospital-shift/experiments/slurm_%j.out

# Runs all planned histopathology-hospital-shift experiments without results (finished ones are skipped; up to three such jobs can
# share the work, see run_batch.py). ORDER=reverse makes a job start from the end of the registry.
R=$HOME/imaging_research
PY=$R/shared_resources/envs/jpr/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export P10_DATA=/var/tmp/${USER}_data/camelyon17_np   # arrays from prepare_hf.py (outside the /home quota)
export WEIGHTS_DIR=${WEIGHTS_DIR:-/var/tmp/${USER}_weights/histopathology-hospital-shift}   # weights kept outside /home; location recorded per run
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
$PY $R/histopathology-hospital-shift/src/run_batch.py $R/histopathology-hospital-shift 70 ${ORDER:-forward}

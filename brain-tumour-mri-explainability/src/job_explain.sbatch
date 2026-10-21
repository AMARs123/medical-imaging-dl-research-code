#!/bin/bash
#SBATCH --job-name=p01_explain
#SBATCH --qos=faculty
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/imaging_research/brain-tumour-mri-explainability/experiments/slurm_explain_%j.out
# Perturbation and robustness tests for every finished run that has none yet. Safe to resubmit.
R=$HOME/imaging_research/brain-tumour-mri-explainability
PY=$HOME/imaging_research/shared_resources/envs/jpr/bin/python
cd $R/src
$PY explain.py $R --device cuda
$PY robust.py $R --device cuda

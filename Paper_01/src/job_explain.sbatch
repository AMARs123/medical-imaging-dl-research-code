#!/bin/bash
#SBATCH --job-name=p01_explain
#SBATCH --qos=faculty
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/Journal_Papers_Research/Paper_01/experiments/slurm_explain_%j.out
# Perturbation and robustness tests for every finished run that has none yet. Safe to resubmit.
R=$HOME/Journal_Papers_Research/Paper_01
PY=$HOME/Journal_Papers_Research/shared_resources/envs/jpr/bin/python
cd $R/src
$PY explain.py $R --device cuda
$PY robust.py $R --device cuda

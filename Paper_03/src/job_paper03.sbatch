#!/bin/bash -l
#SBATCH --job-name=p03_runs
#SBATCH --qos=faculty
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/Journal_Papers_Research/Paper_03/experiments/slurm_%j.out
R=$HOME/Journal_Papers_Research
$R/shared_resources/envs/jpr/bin/python $R/Paper_03/src/run_batch.py $R/Paper_03 ${ORDER:-forward}

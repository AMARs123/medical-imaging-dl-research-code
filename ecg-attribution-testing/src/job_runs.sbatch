#!/bin/bash -l
#SBATCH --job-name=p03_runs
#SBATCH --qos=faculty
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/imaging_research/ecg-attribution-testing/experiments/slurm_%j.out
R=$HOME/imaging_research
$R/shared_resources/envs/jpr/bin/python $R/ecg-attribution-testing/src/run_batch.py $R/ecg-attribution-testing ${ORDER:-forward}

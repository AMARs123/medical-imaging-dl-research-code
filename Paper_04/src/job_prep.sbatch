#!/bin/bash -l
#SBATCH --job-name=p04_prep
#SBATCH --qos=faculty
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/Journal_Papers_Research/Paper_04/logs/prep_%j.out
set -e
R=$HOME/Journal_Papers_Research
$R/shared_resources/envs/jpr/bin/python $R/Paper_04/src/preprocess.py $R/Paper_04

#!/bin/bash -l
#SBATCH --job-name=p04_prep
#SBATCH --qos=faculty
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/imaging_research/chest-xray-pneumonia-shift/logs/prep_%j.out
set -e
R=$HOME/imaging_research
$R/shared_resources/envs/jpr/bin/python $R/chest-xray-pneumonia-shift/src/preprocess.py $R/chest-xray-pneumonia-shift

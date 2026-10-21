#!/bin/bash -l
#SBATCH --job-name=p03_all
#SBATCH --qos=faculty
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/Journal_Papers_Research/Paper_03/logs/all_%j.out
# preprocessing -> 18 training runs -> attribution analysis, in one job (GPU need is small: 1-D models)
set -e
R=$HOME/Journal_Papers_Research
PY=$R/shared_resources/envs/jpr/bin/python
P=$R/Paper_03
export P03_ROOT=$P OMP_NUM_THREADS=2
[ -f $P/data/processed/meta.csv ] || $PY $P/src/preprocess.py $P
$PY $P/src/run_batch.py $P
until grep -q "DOWNLOAD_DONE" $P/logs/download_ptbxlplus_early.log; do sleep 60; done   # PTB-XL+ fiducials
[ -f $P/data/processed/fiducials_raw.npz ] || $PY $P/src/fiducials.py $P
$PY $P/src/attribute.py $P
echo ALL_DONE

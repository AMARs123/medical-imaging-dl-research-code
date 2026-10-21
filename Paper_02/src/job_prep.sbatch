#!/bin/bash -l
#SBATCH --job-name=p02_prep
#SBATCH --qos=faculty
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/Journal_Papers_Research/Paper_02/logs/prep_%j.out

# Preprocessing, dataset inspection and a GPU smoke test of every backbone (2 steps per epoch, 1 epoch).
R=$HOME/Journal_Papers_Research
PY=$R/shared_resources/envs/jpr/bin/python
P=$R/Paper_02
set -e
[ -f $P/data/processed/test_meta.csv ] || $PY $P/src/preprocess.py $P
$PY $P/src/inspect_dataset.py $P
for m in resnet50.a1_in1k efficientnet_b0.ra_in1k convnext_tiny.fb_in1k deit_small_patch16_224.fb_in1k deit3_small_patch16_224.fb_in1k swin_tiny_patch4_window7_224.ms_in1k coatnet_0_rw_224.sw_in1k maxvit_tiny_tf_224.in1k mobilevitv2_100.cvnets_in1k; do
  rm -rf $P/logs/smoke/$m
  $PY $P/src/train.py --proc $P/data/processed --out $P/logs/smoke/$m --model $m --epochs 1 --max-steps 20 --limit 200 --loss la
  $PY -c "import json;r=json.load(open('$P/logs/smoke/$m/results.json'));print('SMOKE', '$m', r['n_params'], round(r['peak_gpu_mb']), round(r['train_seconds'],1))"
done
echo PREP_DONE

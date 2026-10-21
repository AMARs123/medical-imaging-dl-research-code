#!/bin/bash
# waits for both extractions, preprocesses on the login node (CPU, nice 15), smoke-tests train.py on CPU, then queues training
P=$HOME/imaging_research/chest-xray-pneumonia-shift
PY=$HOME/imaging_research/shared_resources/envs/jpr/bin/python
until grep -q "EXTRACT_DONE\|EXTRACT_INCOMPLETE" $P/logs/extract_kermany.log && grep -q NIH_FETCH_DONE $P/logs/fetch_nih.log; do sleep 60; done
if grep -q "EXTRACT_INCOMPLETE" $P/logs/extract_kermany.log $P/logs/fetch_nih.log; then echo "EXTRACTION INCOMPLETE - not submitting"; exit 1; fi
[ -f $P/data/processed/meta.csv ] || nice -n 15 env OMP_NUM_THREADS=2 $PY $P/src/preprocess.py $P > $P/logs/prep.log 2>&1 || { echo PREP_FAILED; exit 1; }
rm -rf $P/logs/smoke
for m in densenet121.tv_in1k convnext_tiny.fb_in1k; do
  nice -n 15 env OMP_NUM_THREADS=2 $PY $P/src/train.py --proc $P/data/processed --out $P/logs/smoke/$m --model $m --train-groups G,NA \
     --epochs 1 --max-steps 2 --limit 40 --device cpu --workers 0 --aug strong >> $P/logs/smoke.log 2>&1 || { echo SMOKE_FAILED $m; exit 1; }
done
grep -q DONE $P/logs/smoke/densenet121.tv_in1k/training.log || { echo SMOKE_FAILED; exit 1; }
sbatch --parsable $P/src/job_runs.sbatch
echo CHAIN_SUBMITTED

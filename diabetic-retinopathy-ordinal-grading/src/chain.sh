#!/bin/bash
# waits for DDR (mirror) and DeepDRiD, unzips, preprocesses (login node, nice 15), smoke-tests on CPU, then queues training
P=$HOME/imaging_research/diabetic-retinopathy-ordinal-grading
PY=$HOME/imaging_research/shared_resources/envs/jpr/bin/python
until grep -q DDR_MIRROR_DONE $P/logs/download_ddr_mirror.log && grep -q DEEPDRID_OK $P/logs/download.log; do sleep 60; done
cd $P/data/raw && [ -d deepdrid ] || { mkdir -p deepdrid && unzip -q deepdrid_master.zip -d deepdrid; }
[ -f $P/data/processed/meta.csv ] || nice -n 15 env OMP_NUM_THREADS=2 $PY $P/src/preprocess.py $P > $P/logs/prep.log 2>&1 || { echo PREP_FAILED; exit 1; }
[ -f $P/experiments/experiment_registry.csv ] || $PY $P/src/make_registry.py $P
rm -rf $P/logs/smoke
for f in corn reg; do
  nice -n 15 env OMP_NUM_THREADS=2 $PY $P/src/train.py --proc $P/data/processed --out $P/logs/smoke/$f --model resnet50.a1_in1k --form $f \
     --epochs 1 --max-steps 2 --limit 30 --device cpu --workers 0 --size 128 >> $P/logs/smoke.log 2>&1 || { echo SMOKE_FAILED $f; exit 1; }
done
sbatch --parsable $P/src/job_runs.sbatch
echo CHAIN_SUBMITTED

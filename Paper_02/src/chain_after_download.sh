#!/bin/bash
# Waits for the download, then submits preprocessing + smoke test, then three training runners that start only if prep succeeded.
P=$HOME/Journal_Papers_Research/Paper_02
until grep -q DOWNLOAD_DONE $P/logs/download.log; do sleep 60; done
J=$(sbatch --parsable $P/src/job_prep.sbatch)
echo "prep job $J"
for o in forward reverse forward; do
  sbatch --parsable --dependency=afterok:$J --export=ALL,ORDER=$o $P/src/job_paper02.sbatch
done
echo CHAIN_SUBMITTED

#!/bin/bash
# Full analysis from stored run outputs (CPU only, light): tables, statistics, figures.
set -e
R=$HOME/Journal_Papers_Research/Paper_01
PY=$HOME/Journal_Papers_Research/shared_resources/envs/jpr/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd $R
nice $PY src/aggregate.py experiments results/processed
nice $PY src/diag_evidence.py . results/statistics/evidence_diagnostics.csv > results/statistics/evidence_diagnostics.txt
nice $PY src/stats.py . > results/statistics/stats_output.txt
nice $PY src/figures.py .
nice $PY src/make_tables.py .
echo ANALYSIS_DONE

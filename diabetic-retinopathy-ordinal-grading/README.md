# Diabetic retinopathy grading: ordinal formulations and uncertainty

**Data.** DDR fundus dataset. The data are public but are not redistributed here; download them from their original sources.

**What the code does.** Nominal vs ordinal formulations of 5-grade DR; sources of uncertainty; evaluation with clinically weighted errors and statistical comparison.

**Code.** All scripts are in `src/`. Typical order: `inspect`/`preprocess` -> `train.py` (one run per experiment ID,
never overwriting a finished run) -> `run_batch.py` (runs all planned experiments) -> analysis/statistics ->
tables/figures. `*.sbatch` files are SLURM jobs for the GPU server; set the project path for your system.

Files: `chain.sh`, `common.py`, `evaluate.py`, `figures.py`, `job_runs.sbatch`, `make_tables_data.py`, `make_tables_results.py`, `preprocess.py`, `run_batch.py`, `stats.py`, `train.py`, `verify_ddr.py`

**Results:** withheld until publication.

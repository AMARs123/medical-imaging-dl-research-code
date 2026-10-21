# Waste sorting: studio-to-wild transfer and edge cost

**Data.** TrashNet (studio images) and TACO (in-the-wild images). The data are public but are not redistributed here; download them from their original sources.

**What the code does.** Compact material classifiers trained on studio images and tested in the wild; accuracy versus latency, size and compute on edge hardware.

**Code.** All scripts are in `src/`. Typical order: `inspect`/`preprocess` -> `train.py` (one run per experiment ID,
never overwriting a finished run) -> `run_batch.py` (runs all planned experiments) -> analysis/statistics ->
tables/figures. `*.sbatch` files are SLURM jobs for the GPU server; set the project path for your system.

Files: `analyze.py`, `data.py`, `efficiency.py`, `extra.py`, `extra2.py`, `extra_q2.py`, `figures.py`, `job_runs.sbatch`, `make_tables_data.py`, `make_tables_extra.py`, `make_tables_results.py`, `preprocess.py`, `run_batch.py`, `stats.py`, `train.py`

**Results:** withheld until publication.

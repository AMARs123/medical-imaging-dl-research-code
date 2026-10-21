# Paper 04: Chest X-ray pneumonia: separating hospital and age shifts

**Data.** NIH ChestX-ray14 and the Kermany paediatric chest X-ray set. The data are public but are not redistributed here; download them from their original sources.

**What the code does.** Pneumonia classifiers evaluated under controlled hospital and age shifts; label-free remedies tested; extra evaluations and statistics.

**Code.** All scripts are in `src/`. Typical order: `inspect`/`preprocess` -> `train.py` (one run per experiment ID,
never overwriting a finished run) -> `run_batch.py` (runs all planned experiments) -> analysis/statistics ->
tables/figures. `*.sbatch` files are SLURM jobs for the GPU server; set the project path for your system.

Files: `analyze.py`, `chain.sh`, `common.py`, `evaluate_extra.py`, `fetch_nih.sh`, `job_paper04.sbatch`, `job_prep.sbatch`, `make_extra.py`, `make_results.py`, `make_tables_data.py`, `preprocess.py`, `run_batch.py`, `select_nih.py`, `train.py`, `verify_nih_mirror.py`

**Results:** withheld until publication.

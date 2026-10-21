# Histopathology: pretraining and stain augmentation under realistic model selection

**Data.** Camelyon17-WILDS. The data are public but are not redistributed here; download them from their original sources.

**What the code does.** Pathology-specific vs generic pretraining and stain augmentation, with model selection on in-distribution data only; seed-to-seed variation and shift to unseen hospitals.

**Code.** All scripts are in `src/`. Typical order: `inspect`/`preprocess` -> `train.py` (one run per experiment ID,
never overwriting a finished run) -> `run_batch.py` (runs all planned experiments) -> analysis/statistics ->
tables/figures. `*.sbatch` files are SLURM jobs for the GPU server; set the project path for your system.

Files: `analyze.py`, `job_runs.sbatch`, `make_results.py`, `make_tables_data.py`, `prepare_hf.py`, `run_batch.py`, `train.py`

**Results:** withheld until publication.

# Skin lesion classification: stability of model rankings

**Data.** HAM10000 (lesion-grouped splits); ISIC 2018 for external testing. The data are public but are not redistributed here; download them from their original sources.

**What the code does.** Convolutional, transformer and hybrid classifiers compared under lesion-grouped evaluation and on an external set; duplicate audit; repeated runs and statistical tests on how stable the model ranking is.

**Code.** All scripts are in `src/`. Typical order: `inspect`/`preprocess` -> `train.py` (one run per experiment ID,
never overwriting a finished run) -> `run_batch.py` (runs all planned experiments) -> analysis/statistics ->
tables/figures. `*.sbatch` files are SLURM jobs for the GPU server; set the project path for your system.

Files: `aggregate.py`, `chain_after_download.sh`, `common.py`, `download_images.sh`, `dup_audit.py`, `figures.py`, `inspect_dataset.py`, `job_runs.sbatch`, `job_prep.sbatch`, `make_tables.py`, `make_tables_results.py`, `pdownload.py`, `preprocess.py`, `run_batch.py`, `stats.py`, `train.py`

**Results:** withheld until publication.

# Segmentation failure detection with uncertainty

**Data.** Kvasir-SEG (polyps) and HAM10000 / ISIC (skin lesions). The data are public but are not redistributed here; download them from their original sources.

**What the code does.** Pixel-level uncertainty aggregated over the predicted foreground to flag failed segmentations; comparison of aggregation rules and uncertainty sources; efficiency.

**Code.** All scripts are in `src/`. Typical order: `inspect`/`preprocess` -> `train.py` (one run per experiment ID,
never overwriting a finished run) -> `run_batch.py` (runs all planned experiments) -> analysis/statistics ->
tables/figures. `*.sbatch` files are SLURM jobs for the GPU server; set the project path for your system.

Files: `analyze.py`, `common.py`, `efficiency.py`, `extra.py`, `figures.py`, `figures_extra.py`, `job_runs.sbatch`, `make_tables.py`, `make_tables_extra.py`, `preprocess.py`, `run_batch.py`, `stats.py`, `train.py`

**Results:** withheld until publication.

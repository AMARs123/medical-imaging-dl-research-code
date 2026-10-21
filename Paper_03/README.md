# Paper 03: ECG attribution maps: reference levels for testing explanations

**Data.** PTB-XL (patient-level folds); Chapman for external testing. The data are public but are not redistributed here; download them from their original sources.

**What the code does.** 12-lead ECG classifiers; attribution maps tested against untrained-network and amplitude baselines; fiducial-point and segment analysis; external evaluation.

**Code.** All scripts are in `src/`. Typical order: `inspect`/`preprocess` -> `train.py` (one run per experiment ID,
never overwriting a finished run) -> `run_batch.py` (runs all planned experiments) -> analysis/statistics ->
tables/figures. `*.sbatch` files are SLURM jobs for the GPU server; set the project path for your system.

Files: `attribute.py`, `attribute_amplitude.py`, `attribute_extra.py`, `efficiency.py`, `eval_external.py`, `fiducials.py`, `figures.py`, `figures_r2.py`, `job_all.sbatch`, `job_paper03.sbatch`, `make_tables.py`, `models.py`, `prepare_external.py`, `preprocess.py`, `run_batch.py`, `s3_download.py`, `segments.py`, `stats.py`, `test_one_external.py`, `train.py`

**Results:** withheld until publication.

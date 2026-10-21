# Paper 07: Glaucoma from fundus photographs: leave-one-camera-out evaluation

**Data.** Chaksu (multi-camera, multi-expert) and PAPILA. The data are public but are not redistributed here; download them from their original sources.

**What the code does.** Leave-one-camera-out evaluation; expert-agreement (soft) targets vs single labels; cross-dataset testing.

**Code.** All scripts are in `src/`. Typical order: `inspect`/`preprocess` -> `train.py` (one run per experiment ID,
never overwriting a finished run) -> `run_batch.py` (runs all planned experiments) -> analysis/statistics ->
tables/figures. `*.sbatch` files are SLURM jobs for the GPU server; set the project path for your system.

Files: `analyze.py`, `analyze_extra.py`, `job_paper07.sbatch`, `make_results.py`, `make_tables_data.py`, `preprocess.py`, `run_batch.py`, `train.py`

**Results:** withheld until publication.

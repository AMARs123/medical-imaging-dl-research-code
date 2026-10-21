# Paper 06: Leaf disease recognition: laboratory-to-field transfer

**Data.** PlantVillage (laboratory) and PlantDoc (field). The data are public but are not redistributed here; download them from their original sources.

**What the code does.** Leaf-grouped evaluation to avoid leakage between images of the same leaf; lab-to-field transfer and the effect of a small number of field labels.

**Code.** All scripts are in `src/`. Typical order: `inspect`/`preprocess` -> `train.py` (one run per experiment ID,
never overwriting a finished run) -> `run_batch.py` (runs all planned experiments) -> analysis/statistics ->
tables/figures. `*.sbatch` files are SLURM jobs for the GPU server; set the project path for your system.

Files: `add_phase_c.py`, `analyze.py`, `build_index.py`, `job_paper06.sbatch`, `make_extra.py`, `make_results.py`, `make_tables_data.py`, `run_batch.py`, `train.py`

**Results:** withheld until publication.

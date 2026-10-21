# Paper 01: Brain tumour MRI: do classifiers look at the tumour?

**Data.** Figshare brain tumour MRI dataset (Cheng et al.). The data are public but are not redistributed here; download them from their original sources.

**What the code does.** Patient-level (identity-aware) cross-validation of CNN and vision-transformer classifiers; explanation maps scored against expert tumour masks; mask-guided evidence (saliency) supervision with ablation, robustness, efficiency and external testing.

**Code.** All scripts are in `src/`. Typical order: `inspect`/`preprocess` -> `train.py` (one run per experiment ID,
never overwriting a finished run) -> `run_batch.py` (runs all planned experiments) -> analysis/statistics ->
tables/figures. `*.sbatch` files are SLURM jobs for the GPU server; set the project path for your system.

Files: `add_lr_check.py`, `add_phase_d.py`, `add_phase_ef.py`, `add_phase_g.py`, `aggregate.py`, `common.py`, `diag_evidence.py`, `efficiency.py`, `eval_external.py`, `explain.py`, `explain_signed.py`, `figures.py`, `figures_extra.py`, `figures_r2.py`, `id_similarity.py`, `inspect_dataset.py`, `job_explain.sbatch`, `job_paper01.sbatch`, `job_paper01_lr.sbatch`, `make_tables.py`, `preprocess.py`, `preprocess_brisc.py`, `robust.py`, `run_analysis.sh`, `run_batch.py`, `stats.py`, `train.py`

**Results:** withheld until publication.

# Trustworthy deep learning for medical and applied imaging: research code

Code for ten studies on how deep-learning image classifiers and segmenters behave under honest evaluation:
patient-, lesion- or leaf-grouped splits, shifts between hospitals, cameras and settings, explanation quality,
uncertainty and edge cost. All studies use public datasets.

> **Note:** the articles are currently being submitted to journals. This repository contains **code only**: no paper,
> no manuscript or other documents, no images or figures, no data and no numerical results. These will be linked
> after publication.

## Studies

| Folder | Topic | Data |
|---|---|---|
| [`Paper_01`](Paper_01) | Brain tumour MRI: do classifiers look at the tumour? | Figshare brain tumour MRI dataset (Cheng et al.) |
| [`Paper_02`](Paper_02) | Skin lesion classification: stability of model rankings | HAM10000 (lesion-grouped splits); ISIC 2018 for external testing |
| [`Paper_03`](Paper_03) | ECG attribution maps: reference levels for testing explanations | PTB-XL (patient-level folds); Chapman for external testing |
| [`Paper_04`](Paper_04) | Chest X-ray pneumonia: separating hospital and age shifts | NIH ChestX-ray14 and the Kermany paediatric chest X-ray set |
| [`Paper_05`](Paper_05) | Diabetic retinopathy grading: ordinal formulations and uncertainty | DDR fundus dataset |
| [`Paper_06`](Paper_06) | Leaf disease recognition: laboratory-to-field transfer | PlantVillage (laboratory) and PlantDoc (field) |
| [`Paper_07`](Paper_07) | Glaucoma from fundus photographs: leave-one-camera-out evaluation | Chaksu (multi-camera, multi-expert) and PAPILA |
| [`Paper_08`](Paper_08) | Waste sorting: studio-to-wild transfer and edge cost | TrashNet (studio images) and TACO (in-the-wild images) |
| [`Paper_09`](Paper_09) | Segmentation failure detection with uncertainty | Kvasir-SEG (polyps) and HAM10000 / ISIC (skin lesions) |
| [`Paper_10`](Paper_10) | Histopathology: pretraining and stain augmentation under realistic model selection | Camelyon17-WILDS |

Each folder has its own README and a `src/` folder with the full pipeline for that study.

## Common design

- **Grouped splits.** Images of the same patient, lesion, leaf or camera are kept in one split, so test results are
  not inflated by near-duplicates.
- **Experiment registry.** Every run has an experiment ID and its own folder (config, command, environment, log,
  metrics, predictions); `train.py` never overwrites a finished run, and `run_batch.py` runs whatever is still
  pending.
- **Multiple seeds and statistics.** Results are aggregated over seeds and folds, with paired statistical tests.
- **External or shifted test sets** wherever an open second dataset exists.

## Setup

```
pip install -r requirements.txt
```

`shared_resources/` holds small helpers used by several studies (plot style, partial download of remote zip files).
The `*.sbatch` files are SLURM jobs for a GPU server; replace `/path/to/Journal_Papers_Research` with your own path.

## Citation

Manuscripts under submission. Citation details will be added after publication. Until then, please contact the
author before reusing this code in a publication.

## Author

Dr. Amar Shukla, IILM University, Gurugram, India.

## License

MIT, see [`LICENSE`](LICENSE). The license covers the code only; each dataset keeps its own license.

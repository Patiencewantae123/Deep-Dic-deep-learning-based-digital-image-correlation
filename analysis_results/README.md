# Deep DIC analysis results

Results from auditing, fixing and test-training this repository on 2 November 2026.
Open `report.html` first: it has the findings, charts and field maps in one page.

These numbers test the pipeline. They do not measure Deep DIC's accuracy. The authors' dataset
and weights are not in the repository, so training used 3,600 synthetic image pairs on a CPU, for
about 30 minutes per network. The fixed notebooks have not yet been run on the authors' data.

## Main results

| Measure | Result |
|---|---|
| Notebooks that run start to finish | 0 of 4 before, 4 of 4 after |
| Behaviour checks on the fixed notebooks | 30 of 30 pass |
| DisplacementNet mean error, 300 unseen pairs | 0.48 px (predicting zero: 1.67 px) |
| Farneback optical flow, same pairs | 0.06 px |
| StrainNet mean error, same pairs | 1.97% (predicting zero: 1.99%), so it did not learn |
| StrainNet on 12 images, overfitting test | loss 99.6% lower, so the pipeline can learn |
| Error without `fcn.eval()` | 1.61 px against 0.47 px |

## Contents

| Path | What it holds |
|---|---|
| `report.html` | Full report |
| `figures/` | Labelled PNG figures |
| `metrics/` | Raw results as JSON |
| `logs/` | Step-by-step training and validation logs |
| `models/` | DisplacementNet weights from the synthetic run |
| `code/` | Scripts that produce everything here |

`models/` has its own `.gitignore`, because the 200 MB weights file exceeds GitHub's 100 MB file
limit. StrainNet's weights are not included, because that run did not learn.

## Figures

| File | Shows |
|---|---|
| `01_training_curves.png` | Validation error for both networks during training |
| `02_test_results.png` | Test error against the baselines |
| `03_silent_bugs.png` | Measured effect of two bugs that give no error message |
| `04_displacement_median_sample.png` | True and predicted displacement, median sample |
| `05_displacement_hard_sample.png` | The same for a difficult sample |
| `06_strain_median_sample.png` | True strain, StrainNet and differentiated displacement |
| `07_strain_overfit_test.png` | StrainNet loss on 12 fixed images |

## Reproducing the results

From `code/`, with the packages in the repository's `requirements.txt`:

```
python gen_data.py 3600
python train.py disp 25 4
python train.py strain 25 4
python evaluate.py
python strain_diagnose.py
```

The synthetic data follows Tables 1 and 2 of the paper. Where the paper leaves details open, the
choices are listed at the top of `code/gen_data.py`.

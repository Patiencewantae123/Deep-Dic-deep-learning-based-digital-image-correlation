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
| ROI tracking, 40-frame synthetic tensile test | committed notebook ends 134 px off the material, fixed notebook 16 px |

## ROI tracking fix

`Diaplacement_run.ipynb` moved the ROI by minus the measured displacement, and moved all four edges by
the same mean value. The paper's data (Section 2.2) moves material at x to x + u, so the ROI has to move
by +u. Its Figure 14 also shows the ROI stretching from 188 × 374 to 178 × 434 px, which a single
mean shift cannot do. The fixed cell flips the sign and moves each edge by the displacement measured
along that edge.

`code/roi_test.py` checks this on 40 frames of a synthetic tensile test (16% vertical stretch, Poisson
contraction, 16 px sideways drift), running cell 4 of each notebook version with the trained
DisplacementNet:

| Version | Worst edge error after 40 frames | Final ROI size (h × w) |
|---|---|---|
| Committed: minus sign, mean shift | 134 px, moving opposite to the material | 366 × 190 |
| Sign flipped only | 40 px | 366 × 190 |
| Fixed: plus sign, each edge follows its own displacement | 16 px | 434 × 185 |
| Fixed, with optical flow in place of the network | 2 px | 421 × 182 |
| Truth | | 425 × 181 |

The optical-flow row shows the test itself is sound. The fixed notebook's remaining 16 px comes from the
briefly trained network, which overestimates the vertical increments by about 20%.

## Longer training (in progress)

`code/train_long.py` continues both networks from the weights above for 120 minutes at learning rate
1e-3 and 25 minutes at 1e-5, both networks at once on the CPU. Logs are written to `logs/long_run/` as
it runs.

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
| `08_roi_tracking.png` | ROI position and edge error over 40 frames, committed against fixed notebook |

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

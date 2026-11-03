# Deep DIC analysis results

Results from auditing, fixing and test-training this repository on 2 November 2026.
Open `report.html` first: it has the findings, charts and field maps in one page.

These numbers test the pipeline. They do not measure Deep DIC's accuracy. The authors' dataset
and weights are not in the repository, so training used 3,600 synthetic image pairs on a CPU: a first
run of about 30 minutes per network, then a longer run to about 2.5 hours per network. The fixed
notebooks have not yet been run on the authors' data.

## Main results

| Measure | Result |
|---|---|
| Notebooks that run start to finish | 0 of 4 before, 4 of 4 after |
| Behaviour checks on the fixed notebooks | 36 of 36 pass |
| DisplacementNet mean error, 300 unseen pairs | 0.48 px after the first run, 0.18 px after the longer run (predicting zero: 1.67 px; paper: 0.038 px) |
| Farneback optical flow, same pairs | 0.06 px |
| StrainNet mean error, same pairs | 1.97% after the first run, 1.63% after the longer run (predicting zero: 1.99%; paper: 0.041%) |
| StrainNet channels learned | 1 of 4: ∂u/∂column, 0.80% against 2.19% for the best constant |
| StrainNet on 12 images, overfitting test | loss 99.6% lower, so the pipeline can learn |
| Error without `fcn.eval()` | 1.61 px against 0.47 px |
| ROI tracking, 40-frame synthetic tensile test | committed notebook ends 127 px off the material, fixed notebook 4 px |

## ROI tracking fix

`Diaplacement_run.ipynb` moved the ROI by minus the measured displacement, and moved all four edges by
the same mean value. The paper's data (Section 2.2) moves material at x to x + u, so the ROI has to move
by +u. Its Figure 14 also shows the ROI stretching from 188 × 374 to 178 × 434 px, which a single
mean shift cannot do. The fixed cell flips the sign and moves each edge by the displacement measured
along that edge.

`code/roi_test.py` checks this on 40 frames of a synthetic tensile test (16% vertical stretch, Poisson
contraction, 16 px sideways drift), running cell 4 of each notebook version with the longer-trained
DisplacementNet:

| Version | Worst edge error after 40 frames | Final ROI size (h × w) |
|---|---|---|
| Committed: minus sign, mean shift | 127 px, moving opposite to the material | 366 × 190 |
| Sign flipped only | 30 px | 366 × 190 |
| Fixed: plus sign, each edge follows its own displacement | 4 px | 418 × 182 |
| Fixed, with optical flow in place of the network | 2 px | 421 × 182 |
| Truth | | 425 × 181 |

The optical-flow row shows the test itself is sound. With the first run's weights the fixed notebook
ended 16 px off (`metrics/roi_test_short_run.json`), because that network overestimated the vertical
increments by about 20%.

## Accumulated fields and strain components

The paper (Figure 2) adds up each frame's increment into accumulated displacement and strain fields,
and reports strain as εxx, εyy and εxy. The run notebooks saved only each frame's increment, and
StrainNet_run saved its four raw gradient channels at half the ROI size. Both now also save:

| File | New key | Contents |
|---|---|---|
| `results/displacement/result_{i}_position1.mat` | `roi_used` | the ROI that frame pair i was measured in (`position` is the updated one) |
| `results/displacement/result_{i}_disp_0_x.mat` | `disp_acc` | accumulated u and v in pixels, on the starting ROI's grid |
| `results/strain/{i}strain.mat` | `strain_xx_yy_xy` | εxx, εyy, εxy for this frame pair, at the ROI's size |
| `results/strain/{i}strain.mat` | `strain_acc` | accumulated εxx, εyy, εxy, on the starting ROI's grid |

The existing keys are unchanged. On the starting ROI's grid, pixel (r, c) is the same material point in
every frame, because the ROI follows the material. Strain is a fraction, as before.
StrainNet_run now crops with `roi_used`, the same ROI DisplacementNet used for that pair. Before, it used
the ROI one frame ahead.

The components come from the four gradient channels: εxx = ∂u/∂column, εyy = ∂v/∂row, and εxy = ½(∂u/∂row +
∂v/∂column). They are corrected for the ROI being resized to multiples of 32. Rigid rotation cancels
in εxy, as the paper requires.

`code/pipeline_test.py` runs both notebooks on the 40-frame tensile test, with optical flow standing in for the
networks, to check this bookkeeping against the known answer:

| Quantity after 40 frames | Notebooks | Truth |
|---|---|---|
| Accumulated displacement, mean error | u 0.21 px, v 0.31 px | v ranges from −82 to −23 px |
| Accumulated εyy | 14.99% | 14.87% |
| Accumulated εxx | −4.75% | −4.92% |
| Accumulated εxy | 0.01% | 0 |

## Differences from the paper left as they are

| Paper | Code | Why it stays |
|---|---|---|
| LeakyReLU (Section 2.1) | ReLU | The authors' released weights are for the code's network. Changing it would stop them loading. |
| Concatenated skip connections (Figure 3) | Added skip connections | Same reason |
| StrainNet outputs εxx, εyy, εxy at full size | 4 gradients at half size | Same reason; StrainNet_run now converts them to the paper's components |
| Loss × 10 (displacement) and × 100 (strain) | Plain MSE | A constant factor on the loss leaves Adam's steps unchanged (apart from its 1e-8 ε), and the strain labels are already in % |

## Repository cleanup

- `results/`: deleted the ten header-only `loglog_*` and `logvalidation_*` files from 29 April. The
  old path-separator bug wrote them there; the notebooks now log to `results/log/`.
- `DisplacementNet_train.ipynb`, `StrainNet_train.ipynb`: cleared the saved output and execution counts. They came from an
  older run on another computer (`c:\Users\HomePC\...`) and did not match the code.
- `README.md`: corrected "PyTorch 1.16" to the paper's 1.6.0, and added how to run the notebooks and what each output file holds.

## Longer training

`code/train_long.py` continued both networks from the first run's weights for 120 minutes at learning
rate 1e-3 and 25 minutes at 1e-5, both at once on the CPU. DisplacementNet reached 3,145 steps
(37,740 images, about one epoch of the paper's dataset) and StrainNet 2,637. Logs are in
`logs/long_run/`, test results in `metrics/eval_long.json`.

| 300 unseen pairs, mean error | Predicting zero | First run | Longer run | Paper's test set |
|---|---|---|---|---|
| DisplacementNet | 1.67 px | 0.48 px | 0.18 px | 0.038 px |
| StrainNet | 1.99% | 1.97% | 1.63% | 0.041% |

StrainNet's improvement comes from one channel. `code/const_baseline.py` compares each channel with
the best single value it could predict everywhere:

| Channel | Best constant | StrainNet | Correlation with labels |
|---|---|---|---|
| ∂u/∂row | 1.80% | 1.81% | −0.01 |
| ∂u/∂column (εxx) | 2.19% | 0.80% | 0.92 |
| ∂v/∂row (εyy) | 2.11% | 2.10% | 0.18 |
| ∂v/∂column | 1.82% | 1.81% | 0.12 |

Both notebooks run with the trained networks on the 40-frame tensile test
(`metrics/pipeline_test_net.json`):

| After 40 frames | Trained networks | Optical flow in their place | Truth |
|---|---|---|---|
| Accumulated displacement, mean error | u 3.4 px, v 3.5 px | u 0.21 px, v 0.31 px | v from −82 to −23 px |
| Accumulated εyy | 6.5% | 15.0% | 14.9% |
| Accumulated εxx | 73.8% | −4.7% | −4.9% |
| Accumulated εxy | 1.1% | 0.0% | 0 |

The accumulated strain is wrong because StrainNet's ∂u/∂column depends on the input size.
`code/strain_probe.py` runs one frame pair, where the truth is −0.12%. StrainNet gives 0.57% on a
128 × 128 crop, 1.16% on 192 × 192 and 1.64% on the notebook's 384 × 192 ROI, and 40 frames add
that up. StrainNet has only seen 128 × 128 inputs. DisplacementNet carries over to larger inputs, as
the ROI test shows. Whether full training removes this needs a GPU run.

## Contents

| Path | What it holds |
|---|---|
| `report.html` | Full report |
| `figures/` | Labelled PNG figures |
| `metrics/` | Raw results as JSON |
| `logs/` | Step-by-step training and validation logs |
| `models/` | DisplacementNet weights: `displacementnet_synthetic.pth` (first run), `displacementnet_synthetic_long.pth` (longer run) |
| `code/` | Scripts that produce everything here |

`models/` has its own `.gitignore`, because each 200 MB weights file exceeds GitHub's 100 MB file
limit. StrainNet's weights are not included, because neither run gives usable strain.

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
| `09_long_training.png` | Validation error over both training runs, against the paper's final value |
| `10_inputs_outputs.png` | What the run notebooks take in and every field they save, on the 40-frame tensile test, against the truth |

## Reproducing the results

From `code/`, with the packages in the repository's `requirements.txt`:

```
python gen_data.py 3600
python train.py disp 25 4
python train.py strain 25 4
python evaluate.py
python strain_diagnose.py
python train_long.py disp 6 120 25 ../logs/long_run      # run these two at the same time
python train_long.py strain 4 120 25 ../logs/long_run
python eval_long.py ../metrics/eval_long.json
python const_baseline.py ../metrics/eval_long.json
python roi_test.py runs/long_disp_phase2.pth ../metrics/roi_test.json roi_before/Diaplacement_run.ipynb 10
python pipeline_test.py flow ../metrics/pipeline_test_flow.json
python pipeline_test.py net ../metrics/pipeline_test_net.json runs/long_disp_phase2.pth runs/long_strain_phase2.pth
python strain_probe.py ../metrics/strain_probe.json
python io_figure.py roi_work/pipeline_net ../figures/10_inputs_outputs.png
python verify_fixed.py ../..
```

`roi_before/` holds `Diaplacement_run.ipynb` as it was before the ROI fix, for the ROI test's comparison.

The synthetic data follows Tables 1 and 2 of the paper. Where the paper leaves details open, the
choices are listed at the top of `code/gen_data.py`.

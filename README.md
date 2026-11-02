# Deep DIC: Deep Learning-Based Digital Image Correlation for End-to-End Displacement and Strain Measurement
 
Note: We are still working on uploading more dataset. <br/>

Digital image correlation (DIC) has become an industry standard to retrieve accurate displacement and strain measurement in tensile testing and other material characterization. Though traditional DIC offers a high precision estimation of deformation for general tensile testing cases, the prediction becomes unstable at large deformation or when the speckle patterns start to tear. In addition, traditional DIC requires a long computation time and often produces a low spatial resolution output affected by filtering and speckle pattern quality. To address these challenges, we propose a new deep learning-based DIC approach – Deep DIC, in which two convolutional neural networks, DisplacementNet and StrainNet, are designed to work together for end-to-end prediction of displacements and strains. DisplacementNet predicts the displacement field and adaptively tracks the region of interest. StrainNet predicts the strain field directly from the image input without relying on the displacement prediction, which significantly improves the strain prediction accuracy. A new dataset generation method is developed to synthesize a realistic and comprehensive dataset, including generation of speckle patterns and deformation of the speckle image with synthetic displacement field. Though trained on synthetic dataset only, Deep DIC is tested on both simulated and experimental data. Deep DIC gives highly consistent and comparable predictions of displacement and strain with those obtained from commercial DIC software, while it outperforms commercial software with very robust strain prediction even at large and localized deformation and varied pattern qualities. In addition, Deep DIC is capable of real-time prediction of deformation with a calculation time down to milliseconds.<br/>
Please refer to our paper: https://www.sciencedirect.com/science/article/pii/S092401362100434<br/>
and https://arxiv.org/ftp/arxiv/papers/2110/2110.13720.pdf

## Citation
If you find this code or the provided data useful in your research, please consider cite:<br/>
@article{yang2022deep,<br/>
title={Deep DIC: Deep learning-based digital image correlation for end-to-end displacement and strain measurement},<br/>
author={Yang, Ru and Li, Yang and Zeng, Danielle and Guo, Ping},<br/>
journal={Journal of Materials Processing Technology},<br/>
volume={302},<br/>
pages={117474},<br/>
year={2022},<br/>
publisher={Elsevier}<br/>
}

## Dependencies
Deep-DIC is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu 20.04, please install PyTorch first following the official instruction. 
- Python 3.7 
- PyTorch (version = 1.6.0, as in the paper)
- Torchvision (version = 0.7.0)
- Pillow (version = 7.2)
- numpy
- scipy
- opencv-python
- CUDA (optional: the notebooks use the GPU when there is one and the CPU otherwise)

The notebooks have also been tested with the newer versions pinned in `requirements.txt` (PyTorch 2.15 on a CPU):
`pip install -r requirements.txt`.

## Overview
We provide:
[https://drive.google.com/drive/folders/13x_-cnnSzsmg1Eiuzg73X7FJSiH69WZE?usp=sharing](https://drive.google.com/drive/folders/13x_-cnnSzsmg1Eiuzg73X7FJSiH69WZE?usp=sharing)
- Datasets: training dataset, validation dataset and test dataset.
- Pre-trained models:
    - DisplacementNet
    - StrainNet
- Code to test with pair of speckle images.
- Code to train the two CNNs with dataset.

## Running the notebooks
Run each notebook from the repository folder. The notebooks create every folder under `results/` themselves.

### Measuring an image sequence
1. Put the frames in `images/`. They are read in file-name order, so number them with leading zeros (`frame_001.png`, `frame_002.png`, ...).
2. Put the pre-trained weights in `results/model/`, named `pretrained_displacementnet` and `pretrained_strainnet`.
3. In `Diaplacement_run.ipynb`, set the starting ROI in the cell with the main loop: `x1, x2, y1, y2` are the top row, bottom row, left column and right column, in pixels. Set `x100, x200, y100, y200` to the same values.
4. Run `Diaplacement_run.ipynb`, then `StrainNet_run.ipynb`. The ROI follows the material from frame to frame, and stretches with it.

Each frame pair `i` (frames `i-1` and `i`) writes:

| File | Key | Contents |
|---|---|---|
| `results/displacement/result_{i}_position1.mat` | `position` | ROI after this pair: top, bottom, left, right, then the height and width used for the pair |
| | `roi_used` | ROI this pair was measured in |
| `results/displacement/result_{i}_disp_0_x.mat` | `disp_1_x` | u and v for this pair, in image pixels, on the ROI resized to multiples of 32 |
| | `disp_acc` | u and v accumulated from the first frame, on the starting ROI's grid |
| `results/strain/{i}strain.mat` | `strain` | StrainNet's 4 raw channels at half size: du/d(row), du/d(column), dv/d(row), dv/d(column) |
| | `strain_xx_yy_xy` | εxx, εyy and εxy for this pair, at the ROI's size |
| | `strain_acc` | εxx, εyy and εxy accumulated from the first frame, on the starting ROI's grid |

Strain is a fraction (0.01 = 1%). On the starting ROI's grid, each pixel is the same material point in every frame.

### Training
1. Put the dataset in `imgs3/` (`train_image_{n}_1.png`, `train_image_{n}_2.png`) and `gt3/` (`train_image_{n}.mat`), next to `train_dataset.txt` and `validation_dataset.txt`.
2. Run `DisplacementNet_train.ipynb` or `StrainNet_train.ipynb`. Both train for 200 epochs, with the learning rate dropping from 1e-3 to 1e-5 after epoch 100, as in the paper.
3. Checkpoints go to `results/model/` and logs to `results/log/`.

A full run takes days on a CPU, so use a GPU. `analysis_results/` holds the audit of this code and its test results.

# UltraG-Ray: Physics-Based Gaussian Ray Casting for Novel Ultrasound View Synthesis

### [Project Page](https://felixduelmer.github.io/publications/ultragray.html) | [Paper (MIDL 2026)](https://openreview.net/forum?id=QIUjZpcGYz)

[Felix Duelmer](https://felixduelmer.github.io/), [Jakob Klaushofer](https://openreview.net/profile?id=~Jakob_Klaushofer1), [Magdalena Wysocki](https://www.cs.cit.tum.de/camp/members/magdalena-wysocki/), [Nassir Navab](https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/), [Mohammad Farid Azampour](https://www.cs.cit.tum.de/camp/members/mohammad-farid-azampour/)

Technical University of Munich

## Code Structure

- **`trainer.py`**: Training script with configuration via CLI arguments or JSON config file
- **`viewer.py`**: Launch viser web viewer and/or benchmarking from a checkpoint
- **`gsplat/`**: Modified base of gsplat
  - **`rendering.py`**: `ultrasound_rasterization()` and `ultrasound_3d_rasterization()` — wrappers for ultrasound rendering and a 3d representation 
  - **`strategy/ultrasound.py`**: `Ultrasound3DStrategy` — strategy for gaussian pruing/splitting for density control
  - **`cuda/csrc/`**: Custom CUDA kernels including:
    - `RasterizeToPixelsUltrasound3DGSFwd.cu` — forward path
    - `RasterizeToPixelsUltrasound3DGSBwd.cu` — backward path 
- **`download_datasets.sh`**: Script to download datasets from Google Drive

## Getting Started

### Prerequisites

- Python >= 3.9
- CUDA-capable GPU
- CUDA toolkit (version must match your PyTorch CUDA build, e.g. CUDA 11.8 with `+cu118`)
- PyTorch
- ninja

### Step 1: Clone the Repository

```sh
git clone <your-repo-url>
cd gsplat
git submodule update --init --recursive
```

The CUDA code depends on the `glm` submodule in `gsplat/cuda/csrc/third_party/glm`.

### Step 2: Create Python Virtual Environment (if you want to)

```sh
python3 -m venv ultragray_env
# or: python -m venv ultragray_env
source ultragray_env/bin/activate
```

### Step 3: Install Dependencies

```sh
pip install --upgrade pip setuptools wheel
pip install -e .
```

This will install all required dependencies, CUDA sources will be compiled later.

If `pip install -e .` fails while building `fused-ssim`:

1. Install a CUDA-matching PyTorch build first (example for CUDA 11.8):
```sh
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
```
2. Retry without build isolation:
```sh
pip install -e . --no-build-isolation
```

### Step 4: Download Datasets (if needed)

Download the porcine muscle and spine phantom datasets:

```sh
./download_datasets.sh
```

This downloads the data into the `data/` directory with the following structure:
```
data/
├── pig_shoulder_v2/
│   ├── images_train.npy    # Training images (N, H, W), grayscale [0, 255]
│   ├── images_val.npy      # Validation images
│   ├── poses_train.npy     # Training poses (N, 4, 4), camera-to-world
│   ├── poses_val.npy       # Validation poses
│   └── conf.json           # Dataset configuration for trainer
└── spine_phantom/
    ├── images_train.npy
    ├── images_val.npy
    ├── poses_train.npy
    ├── poses_val.npy
    └── conf.json
```

New datasets should follow the same structure.

## Reproducing Results

### Training on Porcine Muscle Dataset

```sh
bash train_pig_shoulder.sh
```

Or equivalently:

```sh
python trainer.py --config_file data/pig_shoulder_v2/conf.json
```

### Training on Spine Phantom Dataset

```sh
bash train_spine_phantom.sh
```

Or equivalently:

```sh
python trainer.py --config_file data/spine_phantom/conf.json
```

Results (checkpoints, PLY files, evaluation metrics) are saved to the `result_dir` specified in the config.
For the given datasets this is `results` with subdirectories for the dataset names.
Subsequent runs will overwrite existing data if there are conflicts.

## Using the Web Viewer

The Viewer has two modes:
- Standalone: Via the render mode option you can switch between 3D and ultrasound views.
- Split: Enter split mode by opening the URL inside a second browser window and tile them beside each other. This will lock the render mode option and display 3D and Ultrasound views side-by-side. They are synced, so you will see the position of the probe in 3D in real time and can also move it from the 3D view.

## Visualize Trained Models

Launch the interactive viewer to visualize trained models:

```sh
python viewer.py --ckpt results/pig_shoulder_v2/ckpts/ckpt_29999.pt
```

The viewer provides a web-based interface (via [viser](https://github.com/nerfstudio-project/viser)) for exploring the reconstructed ultrasound scene from arbitrary probe poses.

Use `--benchmark` mode to measure rendering speed:

```sh
python viewer.py --ckpt results/pig_shoulder_v2/ckpts/ckpt_29999.pt --benchmark --benchmark_res_width 512 --benchmark_res_height 512 --config-file data/pig_shoulder_v2/conf.json
```

You must specify config file containing the training/validation poses and probe parameters.
The resolution here doesn't correspond to the actual image proportions, that is defined by the probe parameters in the config file. 
We can render at arbitrary resolutions after training.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{duelmer2026ultragray,
  title={UltraG-Ray: Physics-Based Gaussian Ray Casting for Novel Ultrasound View Synthesis},
  author={Duelmer, Felix and Klaushofer, Jakob and Wysocki, Magdalena and Navab, Nassir and Azampour, Mohammad Farid},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2026}
}
```

## Acknowledgments

This project builds upon [gsplat](https://github.com/nerfstudio-project/gsplat) by the Nerfstudio Team, licensed under the Apache License 2.0.
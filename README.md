# KITTI stereo visual odometry


Stereo visual odometry pipeline using ORB features, StereoBM depth estimation, and PnP pose estimation evaluated against KITTI ground truth. Uses both left and right camera images to recover metric-scale depth — no scale ambiguity unlike monocular VO.
## What it does

1. Loads a KITTI stereo sequence (image_2 + image_3)
2. Computes metric depth per frame via StereoBM disparity
3. Detects ORB keypoints and backprojects them to 3D
4. Matches features across consecutive frames with BFMatcher
5. Estimates relative pose via `solvePnPRansac`
6. Evaluates the trajectory against ground truth using Absolute Trajectory Error (ATE) with Umeyama alignment

## Project structure

```
kitti-stereo-vo/
├── main.py            entry point
├── config.py          all paths and tunable parameters
├── calibration.py     parse calib.txt → K, baseline
├── depth.py           DepthEstimator — StereoBM + backprojection
├── features.py        FeatureMatcher — ORB detect + BF match
├── odometry.py        StereoVO — per-frame PnP loop
├── evaluation.py      GT loading, Umeyama alignment, ATE metrics
├── visualization.py   trajectory and ATE plots
├── results/           output figures (auto-created)
└── requirements.txt
```

## Dataset

Download KITTI odometry from https://www.cvlibs.net/datasets/kitti/eval_odometry.php

You need:
- Stereo grayscale images (22 GB)
- Calibration files
- Ground truth poses

Expected layout:
```
/your/path/kitti-odometry/
├── sequences/
│   └── 00/
│       ├── image_2/   ← left camera
│       ├── image_3/   ← right camera
│       └── calib.txt
└── poses/
    └── 00.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# with defaults from config.py
python main.py

# different sequence, quick debug run
python main.py --sequence 05 --max_frames 500

# skip GT evaluation (if you don't have poses/)
python main.py --no_eval
```

## Sample result (sequence 00)

| Metric   | Value  |
|----------|--------|
| ATE RMSE | ~40 m  |
| ATE mean | ~25 m  |
| ATE max  | ~80 m  |

Values depend on StereoBM parameters and are before any loop closure correction.

## Limitations and next steps

- No loop closure — drift accumulates over the full 4.5 km route of seq 00
- StereoBM produces noisy disparity on textureless regions (sky, road); replacing with StereoSGBM improves depth quality
- No bundle adjustment; adding g2o pose-graph optimisation would reduce drift
- No keyframe selection; processing every frame is redundant at low-speed segments

## Tech stack

- OpenCV 4.x
- NumPy
- Matplotlib

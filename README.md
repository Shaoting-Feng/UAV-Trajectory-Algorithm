# GIPUT: Maximizing Photo Coverage Efficiency for UAV Trajectory
This repository contains the implementation of GIPUT, a method designed to maximize photo coverage efficiency for UAV (Unmanned Aerial Vehicles) trajectories. The project utilizes Stable-Baselines 3 PPO (Proximal Policy Optimization) for training UAV trajectories in a custom OpenAI gym environment, featuring object modeling using Central Angle and analytical geometry calculations.

## Introduction
GIPUT aims to optimize the flight paths of UAVs to ensure maximum photo coverage with minimal energy consumption. This is achieved through advanced reinforcement learning techniques and precise geometric modeling.

## Usage
To train the UAV trajectory model, use the following command:

```bash
python train.py
```

For testing the trained model, use:

```bash
python test.py
```

## Citation
If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{feng2024giput,
  title={GIPUT: Maximizing Photo Coverage Efficiency for UAV Trajectory},
  author={Feng, Shaoting and Li, Qinya and Yang, Yaodong and Wu, Fan and Chen, Guihai},
  booktitle={Asia-Pacific Web (APWeb) and Web-Age Information Management (WAIM) Joint International Conference on Web and Big Data},
  pages={391--406},
  year={2024},
  organization={Springer}
}
```

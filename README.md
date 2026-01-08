# Semi-Supervised-Remote-Sensing-Segmentation-with-Point-Labels

This project implements a **Weakly Supervised Learning** framework for land-cover segmentation using **Point-Level Supervision**. It focuses on the **LoveDA** dataset and uses a **Partial Cross-Entropy Loss** to train models with sparse annotations.

## Features
- **Partial Cross-Entropy Loss**: Handles sparse point labels by ignoring unlabeled pixels.
- **Point Simulation**: helper to simulate sparse point annotations from dense masks.
- **Analysis Tools**: Scripts to visualize class distributions and find specific examples.
- **Experiment Tracking**: Structured saving of model checkpoints.

## Directory Structure
```
├── data_loader.py          # Dataset handling (LoveDA)
├── model.py                # UNet-ResNet18 Architecture
├── loss.py                 # Partial Cross-Entropy Loss
├── train.py                # Main training script
├── analyze_distribution.py # Dataset statistics and finding examples
├── visualize_results.py    # Qualitative evaluation script
├── run_experiments.ps1     # Automation script for experiments
├── report.md               # Technical report
├── result/                 # Checkpoints directory (best_model_exp*.pth)
└── LoveDA/                 # Dataset Directory
```

## Setup
1.  **Dependencies**:
    - PyTorch, Torchvision
    - OpenCV, NumPy, Matplotlib, Tqdm
    - Pandas (optional for advanced analysis)

2.  **Dataset**:
    - Download the LoveDA dataset.
    - Extract it so the structure matches `LoveDA/Train`, `LoveDA/Val`.

## Usage

### 1. Training
Train the model with a specific number of points per class:
```bash
python train.py --points 10 --epochs 10 --exp_name exp1
```
- `--points`: Number of labeled points per class (e.g., 5, 10, 20).
- `--exp_name`: Name suffix for saving the model (saved to `result/best_model_NAME.pth`).

### 2. Evaluation / Visualization
Visualize predictions from a trained model:
```bash
python visualize_results.py --model_path result/best_model_exp1.pth --num_images 3
```
Or use the experiment name shortcut:
```bash
python visualize_results.py --exp_name exp1
```

### 3. Dataset Analysis
Analyze class distribution or find specific examples:
```bash
# Calculate class distribution histogram
python analyze_distribution.py --mode hist --split Train --domain Urban

# Find an example image containing Class 7 (Agriculture)
python analyze_distribution.py --mode find --value 7
```

## Experiments
We conducted experiments varying the sparsity of annotations:
1.  **Sparse (5 points)**: Baseline performance.
2.  **Medium (10 points)**: Optimal trade-off.
3.  **Dense (20 points)**: Diminishing returns.

Run all experiments sequentially using:
# Experiment 1: 5 points per class
python train.py --points 5 --epochs 10 --batch_size 4 --exp_name exp1

# Experiment 2: 10 points per class
python train.py --points 10 --epochs 10 --batch_size 4 --exp_name exp2

# Experiment 3: 20 points per class
python train.py --points 20 --epochs 10 --batch_size 4 --exp_name exp3


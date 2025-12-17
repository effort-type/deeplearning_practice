# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning research project for implementing image classification models using PyTorch. The project contains three main tasks involving different datasets and validation methods. The goal is to compare baseline models (LeNet-5) with advanced architectures not covered in class.

## File Structure

```
deeplearning/
├── CLAUDE.md                        # This file
├── prompt.md                        # Original assignment requirements
├── task1_cats_dogs.ipynb            # Task 1: Cats vs Dogs classification
├── task2_chihuahua_muffin.ipynb     # Task 2: Chihuahua vs Muffin classification
├── task3_fashion_mnist.ipynb        # Task 3: Fashion MNIST (Train 2%: Val 49%: Test 49%)
└── task3_fashion_mnist_v2.ipynb     # Task 3 V2: Fashion MNIST (Train 24.5%: Val 0.5%: Test 50%)
```

## Environment

- **Python Environment**: Conda / Miniconda
- **Framework**: PyTorch 2.0+
- **GPU**: H100 (80GB) with optimizations enabled
- **Output Format**: Jupyter Notebooks (`.ipynb`)

### H100 GPU Optimization Settings

```python
BATCH_SIZE = 128        # 256 for Fashion MNIST (smaller images)
NUM_WORKERS = 8         # Parallel data loading
PIN_MEMORY = True       # Optimized GPU memory transfer
USE_AMP = True          # Mixed Precision (BF16/FP16)

# cuDNN benchmark mode
torch.backends.cudnn.benchmark = True

# New PyTorch 2.0+ API
from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda', enabled=USE_AMP)
with autocast('cuda', enabled=USE_AMP):
    ...
```

## Project Tasks

### Task 1: Cats vs Dogs Classification (`task1_cats_dogs.ipynb`)
- **Dataset**: Kaggle cats-and-dogs-image-classification (128x128 RGB)
- **Validation**: Repeated holdout 5x (train:validation = 3:2)
- **Metrics**: Accuracy, F1 Score (Micro), F1 Score (Macro)
- **Models**: LeNet-5, VGGNet, ResNetCNN, SEResNet, ConvNeXt-Tiny

### Task 2: Chihuahua vs Muffin Classification (`task2_chihuahua_muffin.ipynb`)
- **Dataset**: Kaggle muffin-vs-chihuahua-image-classification (128x128 RGB)
- **Validation**: StratifiedKFold 5-fold cross-validation
- **Metrics**: Accuracy, F1 Score (Micro), F1 Score (Macro)
- **Models**: LeNet-5, VGGNet, ResNetCNN, EfficientNet-Lite, EfficientNet-B0, ConvNeXt-Tiny

### Task 3: Fashion MNIST Classification (`task3_fashion_mnist.ipynb`)
- **Dataset**: PyTorch built-in FashionMNIST (28x28 grayscale, 10 classes)
- **Preprocessing**: Merge train/test, then split (train:validation:test = 2:49:49)
- **Validation**: Repeated holdout 10x
- **Metrics**: Accuracy, F1 Score (Micro), F1 Score (Macro)
- **Models**: LeNet-5, VGGNet, ResNetCNN, SEResNet, ConvNeXt-Small

### Task 3 V2: Fashion MNIST Alternative Split (`task3_fashion_mnist_v2.ipynb`)
- **Dataset**: Same as Task 3 (PyTorch FashionMNIST)
- **Preprocessing**:
  1. Merge train/test (70,000 samples)
  2. Random shuffle
  3. Test = 50% (35,000)
  4. Remaining 50%: Train = 49%, Val = 1%
- **Actual Split**: Train 24.5% (17,150) : Val 0.5% (350) : Test 50% (35,000)
- **Validation**: Repeated holdout 5x
- **Metrics**: Accuracy, F1 Score (Micro)
- **Models**: LeNet-5, VGGNet, ResNetCNN, SEResNet, ConvNeXt-Small

#### Task 3 Version Comparison

| Version | Train | Val | Test | Repeats | Key Difference |
|---------|-------|-----|------|---------|----------------|
| V1 | 2% (1,400) | 49% (34,300) | 49% (34,300) | 10x | Small train, large val |
| V2 | 24.5% (17,150) | 0.5% (350) | 50% (35,000) | 5x | Large train, small val |

## Implemented Models

### 1. LeNet-5 (Baseline, LeCun et al., 1998)
- Structure: `C(5x5) → Tanh → AvgPool → C(5x5) → Tanh → AvgPool → FC`
- Classic CNN taught in class
- Uses Tanh activation and Average Pooling

### 2. VGGNet-style (Simonyan & Zisserman, 2014)
- Structure: `[C(3x3) → BN → ReLU]×n → MaxPool` (repeated)
- Small 3x3 kernels stacked deep
- Modern additions: BatchNorm, ReLU

### 3. ResNetCNN (He et al., 2015)
- Structure: `C(7x7) → BN → ReLU → MaxPool → ResBlock×3 → GAP → FC`
- Residual Connection: H(x) = F(x) + x
- Solves gradient vanishing problem

### 4. SEResNet (Hu et al., 2017)
- Structure: ResNetCNN + SE Block
- SE Block: `GAP → FC → ReLU → FC → Sigmoid → Scale`
- Channel attention mechanism

### 5. EfficientNet-Lite (Custom simplified)
- Structure: `C(3x3) → MBConv×8 → GAP → FC`
- MBConv with SE, 3x3 kernels only
- Lighter than original B0

### 6. EfficientNet-B0 (Tan & Le, 2019)
- Structure: `C(3x3) → MBConv×16 → C(1x1,1280) → GAP → FC`
- Original paper architecture
- Mixed 3x3 and 5x5 kernels, expand_ratio=6

### 7. ConvNeXt-Tiny/Small (Liu et al., 2022)
- Structure: `Stem(4x4,s4) → [ConvNeXtBlock]×{3,3,9,3} → GAP → FC`
- ConvNeXt Block: `DWConv(7x7) → LayerNorm → PWConv(4×expand) → GELU → PWConv → LayerScale → +Residual`
- Key features from Vision Transformer (ViT):
  - Patchify stem: 4×4 conv with stride 4 (like ViT patch embedding)
  - LayerNorm instead of BatchNorm
  - GELU activation instead of ReLU
  - Inverted bottleneck (4× channel expansion)
  - LayerScale for stable training
- Depthwise Conv 7×7 for large receptive field
- Task1/2: ConvNeXt-Tiny (depths=[3,3,9,3], dims=[96,192,384,768])
- Task3: ConvNeXt-Small (depths=[2,2,6], dims=[64,128,256] for 28×28 input)

## Evaluation Metrics

| Metric | Calculation | Use Case |
|--------|-------------|----------|
| **Accuracy** | Correct / Total | Overall performance |
| **F1 (Micro)** | Global TP/FP/FN | = Accuracy for balanced data |
| **F1 (Macro)** | Mean of per-class F1 | Better for class imbalance |

## Implementation Guidelines

1. **Model Architecture**: Store all layers in `self.layers = nn.ModuleList([...])` for visibility
2. **Layer Notation**: Use abbreviations in comments (C=Conv, BN=BatchNorm, R=ReLU, etc.)
3. **Documentation**: Add markdown cells explaining each model's structure and rationale
4. **Code Comments**: Detailed comments for report writing

## Common Commands

```bash
# Activate conda environment
conda activate pytorch

# Install required packages
pip install torch torchvision scikit-learn matplotlib kaggle

# Download Kaggle datasets (requires ~/.kaggle/kaggle.json)
kaggle datasets download -d samuelcortinhas/cats-and-dogs-image-classification
kaggle datasets download -d samuelcortinhas/muffin-vs-chihuahua-image-classification
```

## Known Issues & Solutions

### 1. F1 Micro = Accuracy
- **Issue**: F1 Score (Micro) equals Accuracy for balanced datasets
- **Solution**: Report both Micro and Macro F1 scores

### 2. Deprecated autocast API
```python
# Old (deprecated)
from torch.cuda.amp import autocast, GradScaler
with autocast(enabled=USE_AMP):

# New (PyTorch 2.0+)
from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda', enabled=USE_AMP)
with autocast('cuda', enabled=USE_AMP):
```

### 3. Korean Font in Matplotlib
```python
import matplotlib.font_manager as fm
font_candidates = ['NanumGothic', 'Noto Sans CJK KR', ...]
# Auto-detect available Korean fonts
```

## Training Configuration

| Setting | Task 1 & 2 | Task 3 |
|---------|------------|--------|
| Batch Size | 128 | 256 |
| Image Size | 128x128 RGB | 28x28 Gray |
| Optimizer | AdamW (weight_decay=0.01) | AdamW |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Early Stopping | patience=7 | patience=7 |
| Max Epochs | 30 | 30 |

## Visualization Outputs

Each notebook generates visualization PNG files for easy reporting:

### Task 1 (`task1_`)
- `task1_metrics_comparison.png` - Accuracy, F1 Micro, F1 Macro bar charts
- `task1_grouped_metrics.png` - Grouped bar chart comparing all 3 metrics
- `task1_iteration_f1macro.png` - Per-iteration F1 Macro performance

### Task 2 (`task2_`)
- `task2_metrics_comparison.png` - Accuracy, F1 Micro, F1 Macro bar charts
- `task2_grouped_metrics.png` - Grouped bar chart comparing all 3 metrics
- `task2_fold_f1macro.png` - Per-fold F1 Macro performance

### Task 3 (`task3_`)
- `task3_metrics_comparison.png` - Accuracy, F1 Micro, F1 Macro bar charts
- `task3_grouped_metrics.png` - Grouped bar chart comparing all 3 metrics
- `task3_iteration_f1macro.png` - Per-iteration F1 Macro performance

### Task 3 V2 (`task3_v2_`)
- `task3_v2_metrics_comparison.png` - Accuracy, F1 Micro bar charts
- `task3_v2_grouped_metrics.png` - Grouped bar chart comparing 2 metrics
- `task3_v2_iteration_accuracy.png` - Per-iteration Accuracy performance

## Korean Font Support

All visualization code includes platform-specific Korean font detection:

```python
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    font_list = [f.name for f in fm.fontManager.ttflist
                 if 'Nanum' in f.name or 'Gothic' in f.name]
    if font_list:
        plt.rcParams['font.family'] = font_list[0]
plt.rcParams['axes.unicode_minus'] = False
```

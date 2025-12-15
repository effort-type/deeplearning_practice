# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning research project for implementing image classification models using PyTorch. The project contains three main tasks involving different datasets and validation methods.

## Environment

- **Python Environment**: Conda (configured via VS Code)
- **Framework**: PyTorch
- **Output Format**: Jupyter Notebooks (`.ipynb`)

## Project Tasks

### Task 1: Cats vs Dogs Classification
- Dataset: Kaggle cats-and-dogs-image-classification
- Validation: Repeated holdout (train:validation = 3:2)
- Metrics: Accuracy, F1 Score (Micro)
- Repeat 5 times, report mean and standard deviation

### Task 2: Chihuahua vs Muffin Classification
- Dataset: Kaggle muffin-vs-chihuahua-image-classification
- Validation: 5-fold cross-validation
- Metrics: Accuracy, F1 Score (Micro)
- Report performance for each fold, mean and standard deviation

### Task 3: Fashion MNIST Classification
- Dataset: PyTorch built-in Fashion MNIST
- Preprocessing: Merge train/test, then split (train:validation:test = 2:49:49)
- Validation: Repeated holdout (10 iterations)
- Metrics: Accuracy, F1 Score (Micro)

## Implementation Guidelines

When implementing models for this project:

1. **Model Architecture**: Implement both the main model and a comparison baseline model
2. **Layer Organization**: Store network layers in a list (e.g., `self.layers = [...]`) for clear visibility of the architecture
3. **Model Choice**: Use novel architectures not covered in class to improve performance
4. **Documentation**: Add markdown cells and detailed code comments in notebooks for report writing

## Common Commands

```bash
# Activate conda environment
conda activate <env_name>

# Install PyTorch (if needed)
conda install pytorch torchvision -c pytorch

# Download Kaggle datasets (requires kaggle CLI)
kaggle datasets download -d samuelcortinhas/cats-and-dogs-image-classification
kaggle datasets download -d samuelcortinhas/muffin-vs-chihuahua-image-classification
```

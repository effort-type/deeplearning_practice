# Repository Guidelines

## Project Structure & Module Organization
- Root notebooks drive the work: `task1_cats_dogs.ipynb`, `task2_chihuahua_muffin.ipynb`, `task3_fashion_mnist.ipynb`. `prompt.md` is the assignment brief; `CLAUDE.md` records model guidance; `.vscode/` holds editor prefs.
- Keep datasets and artifacts out of git (e.g., `data/cats_dogs/`, `data/muffin_chihuahua/`, `data/fashion_mnist/`); store processed tensors next to the task notebook when needed.
- Structure notebooks consistently: imports → config/constants → data prep → model definitions → training → evaluation/reporting.

## Build, Test, and Development Commands
- Create/activate an environment: `conda create -n dl python=3.10 && conda activate dl`.
- Install core deps: `pip install torch torchvision scikit-learn matplotlib kaggle jupyter`.
- Launch notebooks: `jupyter lab` (or `jupyter notebook`) from the repo root.
- Download data (requires `~/.kaggle/kaggle.json`):  
  `kaggle datasets download -d samuelcortinhas/cats-and-dogs-image-classification`  
  `kaggle datasets download -d samuelcortinhas/muffin-vs-chihuahua-image-classification`
- Optional sanity check for GPU:  
  `python - <<'PY'\nimport torch; print(torch.__version__, torch.cuda.is_available())\nPY`

## Coding Style & Naming Conventions
- Python 3.10+, PEP8, 4-space indentation; `snake_case` for variables/functions, `PascalCase` for classes.
- Show model structure with `self.layers = nn.ModuleList([...])` and add short markdown cells explaining architecture choices over LeNet-5.
- Control randomness: set seeds and log split keys; enable `torch.backends.cudnn.benchmark = True` only when input shapes are fixed.
- Naming: dataloaders `train_loader/val_loader/test_loader`, metrics `acc`, `f1_micro`, `f1_macro`, checkpoints `best_model.pth`.

## Testing Guidelines
- No automated suite; validate by executing notebooks. Ensure evaluation cells report mean and std over required repeats/folds (Task 1: 5× holdout, Task 2: StratifiedKFold 5×, Task 3: 10× holdout with 2:49:49 split).
- When changing data or models, re-run with a second seed to confirm stability; log the seed and data split recipe.
- Do not commit large outputs; keep checkpoints local and capture final metrics in markdown tables.

## Commit & Pull Request Guidelines
- Commit messages: concise, present-tense (e.g., `Add efficientnet-lite for muffin task`); keep changes scoped.
- PR checklist: summary of notebook edits, data source or download command used, metrics table (accuracy + F1 micro/macro), environment notes (torch version, GPU/CPU), and training caveats (epochs, augmentations, seeds). Add plots only when they clarify results.
- Do not add datasets, checkpoints (`*.pth`/`*.pt`), or notebook checkpoints (`.ipynb_checkpoints/`) to git; add ignore entries if new artifacts appear.

## Security & Data Handling
- Keep Kaggle credentials in `~/.kaggle/kaggle.json` with 600 permissions; never commit tokens or dataset zips.
- When sharing notebooks, clear personal paths/usernames and strip bulky outputs unless they capture key results.

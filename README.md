# PharmaPrompt

PharmaPrompt is a repository for molecular property training and attribution analysis.

This public release keeps code and source datasets in GitHub, while model weights are distributed separately through Google Drive.


## Model Download

The model weights and task-specific checkpoints:

- `pharmaprompt_shared_models.zip`
  https://drive.google.com/open?id=1d2LcD0o20lgS29cMaDQkC_E8q0JT_aYA
- `pharmaprompt_chembert_den1.zip`
  https://drive.google.com/open?id=1CSalCaThqyiedeGg0HR-oJVCIC1OyU87
- `pharmaprompt_example_task_checkpoints.zip`
  https://drive.google.com/open?id=13xNV3TebAGDxsO4cNWYlmwXqXPpDmzb_

After downloading, the repository should contain at least these paths:

```text
pretrained/base/base.pth
pretrained/BiomedBERT/config.json
pretrained/BiomedBERT/vocab.txt
pretrained/BiomedBERT/pytorch_model.bin
checkpoints/DEN1/pytorch_model.bin
save/<YOUR_TASK>/.../best_model.pth
```

If you package the files as zip archives for Google Drive, unzip them at the repository root so the relative paths remain unchanged.

Quick local check:

```bash
test -f pretrained/base/base.pth
test -f pretrained/BiomedBERT/pytorch_model.bin
test -f checkpoints/DEN1/pytorch_model.bin
```

The `DEN1` checkpoint is only required if you want to support code paths that instantiate the chembert SMILES encoder.

## Environment

The recommended runtime is the `PharmAgent` conda environment.

```bash
conda activate PharmAgent
```

If you are reproducing the environment from scratch, the repository provides two dependency entry points:

- `environment.yml`: the recommended conda base environment for Python, RDKit, and the scientific Python stack
- `requirements.txt`: additional Python packages used across scripts

Recommended installation order:

```bash
conda env create -f environment.yml
conda activate PharmAgent
pip install -r requirements.txt
```

The provided environment files cover the lightweight and common dependencies used by most scripts, including:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `networkx`
- `requests`
- `lmdb`
- `rdkit`
- `transformers`
- `ogb`
- `tdc`
- `python-Levenshtein`
- `pandas-flavor`
- `plip`

Optional server-related dependencies are also included:

- `fastapi<0.100`
- `uvicorn[standard]`
- `pydantic<2`

Heavy GPU-sensitive packages are intentionally not pinned in `requirements.txt` because installation depends on your CUDA, PyTorch, and platform combination. Install these separately using the official instructions for your system:

- `torch`
- `dgl`
- `torch-geometric`
- `dgllife`

Check the core dependencies:

```bash
python -c "import torch, dgl, rdkit, shap; print('torch', torch.__version__); print('dgl_ok'); print('rdkit_ok'); print('shap_ok')"
```

For a minimal smoke test after environment setup, you can also run:

```bash
python -c "import numpy, pandas, scipy, sklearn, transformers, lmdb; print('basic_python_stack_ok')"
```

## Data

The `datasets/` directory in this repository already contains the source CSV datasets.

Use the following entry point to preprocess a raw dataset:

```bash
python scripts/prepare_raw_dataset.py \
  --input_csv "/abs/path/to/your_raw.csv" \
  --dataset_name YOUR_DATASET \
  --dataset_kind ligand
```

Notes:

- The input CSV must contain at least one `smiles` column. `SMILES` and `Smiles` are also accepted.
- All remaining columns are treated as supervision targets.
- For virtual screening data, change `--dataset_kind` to `vs`.

## Training

Inspect the training arguments first:

```bash
python scripts/finetune_pharmaPrompt.py --help
```

Minimal training example:

```bash
python scripts/finetune_pharmaPrompt.py \
  --mode finetune \
  --dataset HPK1_IC50 \
  --data_path "$(pwd)/datasets/ligand" \
  --device cuda:1 \
  --num_runs 1 \
  --num_workers 0 \
  --n_epochs 1 \
  --batch_size 4 \
  --use_amp true
```

This workflow generates derived training artifacts such as:

- `splits/scaffold-*.npy`
- `<dataset>_5.pkl`
- `rdkfp1-7_512.npz`
- `molecular_descriptors.npz`
- `phar_features_lmdb/`
- `smiles_embeddings_chembert_lmdb/`

## Attribution

Inspect the available script arguments first:

```bash
python scripts/attribution_deepshap_single.py --help
```

For attribution you typically need:

- `pretrained/base/base.pth`
- `pretrained/BiomedBERT/*`
- One task-specific checkpoint under `save/.../best_model.pth`

For chembert-based SMILES encoder paths, also provide:

- `checkpoints/DEN1/pytorch_model.bin`

### Model Evaluation

```bash
python scripts/finetune_pharmaPrompt.py \
  --mode evaluate \
  --dataset HPK1_IC50 \
  --data_path "$(pwd)/datasets/ligand" \
  --device cuda:3 \
  --num_runs 1 \
  --num_workers 0 \
  --batch_size 4
```

### Single-Molecule DeepSHAP Attribution

The attribution script remains available, but the current public Google Drive release does not ship a dedicated attribution example checkpoint.

If you already have a compatible finetuned checkpoint locally, pass it explicitly:

```bash
python scripts/attribution_deepshap_single.py \
  --model YOUR_TASK_NAME \
  --smiles CCO \
  --background_n 2 \
  --ckpt_path /abs/path/to/your/best_model.pth
```

If you only want to test the script logic without loading a model, run:

```bash
python scripts/attribution_deepshap_single.py --test_aggregation_only
```

## Google Drive Packaging Recommendation

For the current public release, keep Google Drive organized in three layers:

- Shared base models: `pretrained/base/` and `pretrained/BiomedBERT/`
- ChemBERT support: `checkpoints/DEN1/pytorch_model.bin`
- Task checkpoints: upload only the task directories you need for evaluation or attribution

The current release uses these archive names:

- `pharmaprompt_shared_models.zip`
- `pharmaprompt_chembert_den1.zip`
- `pharmaprompt_example_task_checkpoints.zip`

The current practical release that covers the documented public workflows is:

- shared base weights
- `DEN1` if you want chembert path support
- one compatible finetuned checkpoint for the task you want to evaluate or attribute

If you want broader public task coverage later, add more `save/<TASK>/.../best_model.pth` files following the same relative layout.

Additional checkpoints can be published later as separate follow-up releases.

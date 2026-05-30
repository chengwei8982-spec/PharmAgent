# PharmAgent

- Manuscript title: PharmAgent: An Interpretable Pharmacophore-Guided Reasoning Agent for Drug Discovery
- Authors: Chengwei Ai, Yufeng Mao, Yuqing Su, Qiaozhen Meng, Shiqiang Ma, Cheng Liang, Housheng Su, Qianqian Yuan, Xiaoyi Liu, Fei Guo

PharmAgent is designed as a pharmacophore-guided reasoning agent for drug discovery. The manuscript describes a framework that formulates pharmacophore hypotheses, answers pharmacophore queries instead of only encoding molecules, and returns interpretable pharmacophore-level evidence together with predictive results.


PharmAgent is the code repository for PharmAgent: An Interpretable Pharmacophore-Guided Reasoning Agent for Drug Discovery.

The project centers on pharmacophore-guided reasoning, aiming to provide predictive performance together with pharmacophore-level evidence that can be inspected in drug discovery workflows.

This public release keeps code and source datasets in GitHub, while model weights are distributed separately through Google Drive.


## Model Download

The following Google Drive folder that contains the required pretrained weights and task checkpoints:

- Shared models archive
  https://drive.google.com/open?id=1d2LcD0o20lgS29cMaDQkC_E8q0JT_aYA
- DEN1 support archive
  https://drive.google.com/file/d/1CSalCaThqyiedeGg0HR-oJVCIC1OyU87/view?usp=sharing
- Task checkpoints archive
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

## System Requirements

Validated environment for the current public release:

- Operating system: Ubuntu 20.04.2 LTS (validated on the current server)
- Python: 3.7.13
- Core packages tested in the validated environment: PyTorch 1.13.1+cu117, DGL 0.9.1.post1, DGLLife 0.2.9, PyTorch Geometric 2.3.1, RDKit 2023.03.2, Transformers 4.30.0, SHAP 0.42.1, pandas 1.3.5, NumPy 1.21.5
- Non-standard hardware: a CUDA-capable NVIDIA GPU is recommended for the validated training, evaluation, and attribution workflows; CPU runs are supported for small smoke tests and portability checks

The codebase is organized as Python source code rather than compiled standalone software. Linux is the validated operating system for the current release.

## Environment

Use the  `PharmAgent` environment file:

```bash
conda env create -f environment.pharmagent.yml
# Important: this environment must install the CUDA 11.7 PyTorch wheels, not the default CPU wheels.
# If you install torch manually, use the cu117 index so pip does not fall back to CPU builds.
conda activate PharmAgent
```

If `conda env create -f environment.pharmagent.yml` fails on your machine, install with a minimal conda base and then install pip packages manually:

```bash
# 0) if mamba is not installed, install it in base
conda install -n base -c conda-forge mamba -y

# 1) create minimal base
mamba create -n PharmAgent python=3.7.13 pip cudatoolkit=11.7 pytorch-mutex=1.0=cuda -c pytorch -c conda-forge -c defaults --channel-priority flexible -y
conda activate PharmAgent

# 2) install the validated pip package set
python -m pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  --find-links https://data.pyg.org/whl/torch-1.13.0+cu117.html \
  --find-links https://data.dgl.ai/wheels/cu117/repo.html \
  torch==1.13.1+cu117 \
  torchaudio==0.13.1+cu117 \
  dgl==0.9.1 dgllife==0.2.9 \
  torch-geometric==2.3.1 \
  torch-scatter==2.1.1+pt113cu117 \
  torch-sparse==0.6.17+pt113cu117 \
  torch-cluster==1.6.1+pt113cu117 \
  torch-spline-conv==1.2.2+pt113cu117 \
  numpy==1.21.6 scipy==1.7.3 pandas==1.3.5 lmdb==1.4.1 \
  rdkit-pypi==2022.9.5 \
  huggingface-hub==0.16.4 safetensors==0.3.1 transformers==4.30.0 \
  python-Levenshtein==0.12.2 pandas-flavor==0.2.0 \
  tensorboard==2.11.2 PyTDC==0.4.1 \
  shap==0.41.0
```

Important notes:

- The environment includes the training, evaluation, and attribution dependencies used in the validated server setup, including `torch`, `dgl`, `torch-geometric`, `dgllife`, and `shap`.
- `tensorboard` and `PyTDC` are required by the evaluate pipeline (`scripts/finetune_pharmagent.py`).
- If the environment is created by `mamba`/`micromamba`, prefer `mamba run -n PharmAgent ...` for non-interactive commands.

Check the core dependencies:

```bash
python -c "import torch, dgl, rdkit, shap; print('torch', torch.__version__); print('dgl_ok'); print('rdkit_ok'); print('shap_ok')"
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

This workflow generates derived training artifacts such as:

- `splits/scaffold-*.npy`
- `<dataset>_5.pkl`
- `rdkfp1-7_512.npz`
- `molecular_descriptors.npz`
- `phar_features_lmdb/`
- `smiles_embeddings_chembert_lmdb/`

Notes:

- The input CSV must contain at least one `smiles` column. `SMILES` and `Smiles` are also accepted.
- All remaining columns are treated as supervision targets.
- For virtual screening data, change `--dataset_kind` to `vs`.

### Data Split

After preprocessing your dataset, generate the split files required by training and evaluation. For scaffold-based ligand and benchmark workflows, use the split generation script below:

```bash
for seed in 0 1 2; do
  python scripts/preprocess_scaffold_split.py \
    --root_path datasets/ligand \
    --use_split_method random_scaffold_split \
    --dataset YOUR_DATASET \
    --seed "$seed"
done
```

This creates the `splits/` files consumed by the `scaffold` finetune and evaluate modes.

If you want to reuse the prepared split definitions used for the paper datasets instead of generating them locally, download the split archive here:

- `dataset_split_files.tar.gz`: https://drive.google.com/open?id=1bzNYMBeJSzo1WXKbzOy51wjg441SYjW_

After downloading, extract the archive at the repository root so the `datasets/` tree is restored in the layout expected by the codebase. 

## Training

Inspect the training arguments first:

```bash
python scripts/finetune_pharmagent.py --help
```

Minimal training example:

```bash
python scripts/finetune_pharmagent.py \
  --mode finetune \
  --dataset HPK1_IC50 \
  --data_path "$(pwd)/datasets/ligand" \
  --device cuda:1 \
  --num_runs 1 \
  --num_workers 0 \
  --n_epochs 1 \
  --batch_size 4 \
```


After finetuning, checkpoints are saved under:

`save/<DATASET>/question_num8/scaffold-<SPLIT>/seed_<SEED>/text_model_pubmed/base_encoder_LiGhT/alpha_0.1_beta_0.1/`

Use the finetuned checkpoint below for evaluation/attribution:

`save/<DATASET>/question_num8/scaffold-<SPLIT>/seed_<SEED>/text_model_pubmed/base_encoder_LiGhT/alpha_0.1_beta_0.1/best_model.pth`

### Model Evaluation

Before running this step, either download the required model files from the Model Download section or replace the path below with your own finetuned checkpoint.

Use the finetuned checkpoint from your training output (replace scaffold/seed as needed):

```bash
python scripts/finetune_pharmagent.py \
  --mode evaluate \
  --dataset HPK1_IC50 \
  --data_path "$(pwd)/datasets/ligand" \
  --device cuda:0 \
  --num_runs 1 \
  --num_workers 0 \
  --batch_size 4 \
  --model_path "$(pwd)/save/<DATASET>/question_num8/scaffold-0/seed_42/text_model_pubmed/base_encoder_LiGhT/alpha_0.1_beta_0.1/best_model.pth"
```

### Tiny Smoke Test

This section is the demo workflow for the current public release.

Expected demo outputs:

- `workspace/HPK1_IC50_tiny.csv`
- `workspace/HPK1_IC50_tiny/` with graph, descriptor, LMDB, and split artifacts
- console metrics including `Test_spear` and `Test_pear`


1. Create a small CSV from the full HPK1 source file.

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv('datasets/ligand/HPK1_IC50/HPK1_IC50.csv')
df.head(64).to_csv('workspace/HPK1_IC50_tiny.csv', index=False)
print('wrote workspace/HPK1_IC50_tiny.csv')
PY
```

2. Precompute graph, fingerprint, and descriptor features for the tiny dataset.

```bash
python scripts/prepare_raw_dataset.py \
  --input_csv "$(pwd)/workspace/HPK1_IC50_tiny.csv" \
  --dataset_name HPK1_IC50_tiny \
  --dataset_kind ligand \
  --output_root "$(pwd)/workspace" \
  --graph_n_jobs 4
```

3. Generate the extra split files required by the `scaffold` finetune and evaluate mode.

```bash
for seed in 0 1 2; do
  python scripts/preprocess_scaffold_split.py \
    --root_path workspace \
    --use_split_method random_scaffold_split \
    --dataset HPK1_IC50_tiny \
    --seed "$seed"
done
```

4. Build the LMDB features required by training and evaluation.

```bash
python - <<'PY'
from scripts.prepare_raw_dataset import build_feature_lmdbs

build_feature_lmdbs(
    dataset_dir='workspace/HPK1_IC50_tiny',
    csv_path='workspace/HPK1_IC50_tiny/HPK1_IC50_tiny.csv',
)
PY
```

5. Run a tiny evaluation smoke test with the shared base checkpoint.

Download the HPK1 example checkpoint here if you want to reproduce this smoke test directly:

- `HPK1.zip`: https://drive.google.com/open?id=1y7FbJcC_iFme_gjXB1izsbD902XO-bAU

After downloading, extract it at the repository root path. You can also replace `--model_path` with your own trained checkpoint.

```bash
python scripts/finetune_pharmagent.py \
  --mode evaluate \
  --dataset HPK1_IC50_tiny \
  --data_path "$(pwd)/workspace" \
  --split_method scaffold \
  --device cpu \
  --num_runs 1 \
  --num_workers 0 \
  --batch_size 8 \
  --model_path "$(pwd)/save/HPK1_IC50/question_num8/scaffold-3/seed_42/text_model_pubmed/base_encoder_LiGhT/alpha_0.1_beta_0.1/best_model.pth" \
  --alpha 0.1 \
  --beta 0.1
```

This tiny evaluation path was validated on the current server and produced `Test_spear` / `Test_pear` outputs across the scaffold evaluation splits.


### DeepSHAP Attribution

The attribution script remains available, but the current public Google Drive release does not ship a dedicated attribution example checkpoint.

Before running this step, replace `--ckpt_path` with a compatible finetuned checkpoint from your own training output, or with a downloaded task checkpoint that matches the target model.

If you already have a compatible finetuned checkpoint locally, pass it explicitly.
Use `--device cpu` for a portable run (recommended on machines without DGL CUDA backend).
If you want GPU attribution, ensure CUDA-enabled DGL is installed and pass a CUDA device (for example, `--device cuda:0`).

```bash
python scripts/attribution_deepshap_single.py \
  --model YOUR_TASK_NAME \
  --smiles CCO \
  --device cpu \
  --background_n 2 \
  --ckpt_path "$(pwd)/save/HPK1_IC50/question_num8/scaffold-3/seed_42/text_model_pubmed/base_encoder_LiGhT/alpha_0.1_beta_0.1/best_model.pth"
```

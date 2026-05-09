# PharmaPrompt

PharmaPrompt is a repository for molecular property training, inference, and attribution analysis.

This public release keeps code and source datasets in GitHub, while model weights are distributed separately through Google Drive.

## Release Layout

- GitHub includes: source code, scripts, server code, source datasets, and dependency files.
- Google Drive includes: pretrained weights and finetuned task checkpoints.
- To run inference, attribution, or the server, download the required model files from Google Drive and place them under the repository root with the original relative paths preserved.

The current public Google Drive release is documented in [MODEL_RELEASE_MINIMAL.md](MODEL_RELEASE_MINIMAL.md), including the optional `DEN1` checkpoint for chembert-based paths.

## Model Download

Prepare a Google Drive folder that contains the files listed in [MODEL_RELEASE_MINIMAL.md](MODEL_RELEASE_MINIMAL.md).

The current public Google Drive release is organized as three archives:

- `pharmaprompt_shared_models.zip`
- `pharmaprompt_chembert_den1.zip`
- `pharmaprompt_public_task_checkpoints_core4.zip`

After downloading, the repository should contain at least these paths:

```text
pretrained/base/base.pth
pretrained/BiomedBERT/config.json
pretrained/BiomedBERT/vocab.txt
pretrained/BiomedBERT/pytorch_model.bin
checkpoints/DEN1/pytorch_model.bin
save/EGFR/.../best_model.pth
save/HPK1_IC50/.../best_model.pth
save/FGFR1_IC50/.../best_model.pth
save/JAK1/.../best_model.pth
```

If you package the files as zip archives for Google Drive, unzip them at the repository root so the relative paths remain unchanged.

The four-task public checkpoint archive currently includes:

- `EGFR`
- `HPK1_IC50`
- `FGFR1_IC50`
- `JAK1`

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

## Testing And Inference

Inspect the available script arguments first:

```bash
python scripts/predict_approved.py --help
python scripts/predict_drugbank_top20.py --help
python scripts/attribution_deepshap_single.py --help
```

For almost all inference workflows you need:

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

### Approved Drug Prediction

This example expects the EGFR finetuned checkpoint from Google Drive:

```bash
python scripts/predict_approved.py \
  --model_name EGFR \
  --approved_name EGFR_approved_2 \
  --device cuda:0 \
  --batch_size 16 \
  --num_workers 0
```

Required checkpoint path:

```text
save/EGFR/question_num8/scaffold-0/seed_42/text_model_pubmed/base_encoder_LiGhT/alpha_0.1_beta_0.1/best_model.pth
```

### DrugBank Top-20 Screening

This example expects the HPK1_IC50 finetuned checkpoint from Google Drive:

```bash
python scripts/predict_drugbank_top20.py \
  --checkpoint_dir save/HPK1_IC50/question_num8/scaffold-3/seed_42/text_model_pubmed/base_encoder_LiGhT/alpha_0.1_beta_0.1 \
  --source_drugbank_csv /abs/path/to/drugbank.csv \
  --source_smiles_col SMILES \
  --train_dataset_name HPK1_IC50 \
  --train_split_name scaffold-3 \
  --device cuda:0
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
- Task checkpoints: upload only the task directories you want to support publicly

The current release uses these archive names:

- `pharmaprompt_shared_models.zip`
- `pharmaprompt_chembert_den1.zip`
- `pharmaprompt_public_task_checkpoints_core4.zip`

The current practical release that covers the documented public tasks is:

- shared base weights
- `DEN1` if you want chembert path support
- `EGFR` checkpoint for `predict_approved.py`
- `HPK1_IC50` checkpoint for `predict_drugbank_top20.py`
- `FGFR1_IC50` checkpoint for additional public inference coverage
- `JAK1` checkpoint for additional public inference coverage

If you want broader public inference coverage later, add more `save/<TASK>/.../best_model.pth` files following the same relative layout.

Additional checkpoints such as `bbbp`, `clintox`, `sider`, `tox21`, `toxcast`, and individual `CHEMBL*` tasks can be published later as separate follow-up releases.

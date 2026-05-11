# PharmAgent

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

## Environment

Use the  `PharmAgent` environment file:

```bash
conda env create -f environment.pharmagent.yml
conda activate PharmAgent
```

Important notes:

- `environment.pharmagent.yml` is the closest match to the environment that was actually used to validate preprocessing, evaluation, and attribution on the current server.
- The environment includes the training, evaluation, and attribution dependencies used in the validated server setup, including `torch`, `dgl`, `torch-geometric`, `dgllife`, and `shap`.

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

Notes:

- The input CSV must contain at least one `smiles` column. `SMILES` and `Smiles` are also accepted.
- All remaining columns are treated as supervision targets.
- For virtual screening data, change `--dataset_kind` to `vs`.

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

This workflow generates derived training artifacts such as:

- `splits/scaffold-*.npy`
- `<dataset>_5.pkl`
- `rdkfp1-7_512.npz`
- `molecular_descriptors.npz`
- `phar_features_lmdb/`
- `smiles_embeddings_chembert_lmdb/`

### Tiny HPK1 Smoke Test

The following commands create and validate a tiny HPK1 dataset under `workspace/` without touching the full dataset under `datasets/ligand/`.

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
for seed in 4 5; do
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
  --model_path "$(pwd)/pretrained/base/base.pth" \
  --alpha 0.1 \
  --beta 0.1
```

This tiny evaluation path was validated on the current server and produced `Test_spear` / `Test_pear` outputs across the scaffold evaluation splits.


### Model Evaluation

```bash
python scripts/finetune_pharmagent.py \
  --mode evaluate \
  --dataset HPK1_IC50 \
  --data_path "$(pwd)/datasets/ligand" \
  --device cuda:3 \
  --num_runs 1 \
  --num_workers 0 \
  --batch_size 4 \
  --model_path "$(pwd)/pretrained/base/base.pth"
```

### DeepSHAP Attribution

The attribution script remains available, but the current public Google Drive release does not ship a dedicated attribution example checkpoint.

If you already have a compatible finetuned checkpoint locally, pass it explicitly:

```bash
python scripts/attribution_deepshap_single.py \
  --model YOUR_TASK_NAME \
  --smiles CCO \
  --background_n 2 \
  --ckpt_path /abs/path/to/your/best_model.pth
```

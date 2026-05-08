# PharmaPrompt（已实测命令）

当前 GitHub 仓库按代码优先发布：默认不包含大体积模型权重，例如 `pretrained/base/base.pth` 和各任务目录下的 `best_model.pth`。

- 训练可直接基于本仓库代码和你自己的数据重新生成权重。
- `attribution_deepshap_single.py` 运行时需要通过 `--ckpt_path` 提供 finetuned checkpoint；若未提供，也可按脚本默认规则在本地已有 `save/.../best_model.pth` 时运行。

以下命令均已在本机 **`KPGT`** 环境中实际验证。

## 环境

先进入 `KPGT` 环境：

```bash
conda activate KPGT
```

依赖检查：

```bash
python -c "import torch, rdkit, dgl; print('torch', torch.__version__); print('rdkit_ok'); print('dgl_ok')"
```

## 训练

### 1) 原始 CSV 预处理为可训练数据

原始 CSV 至少需要一列：

- `smiles`（或 `SMILES` / `Smiles`）

其余列会被当作监督标签。

下面这个流程已在 `KPGT` 环境中实际跑通，会自动生成：

- `splits/scaffold-*.npy`
- `<dataset>_5.pkl`
- `rdkfp1-7_512.npz`
- `molecular_descriptors.npz`
- `phar_features_lmdb/`
- `smiles_embeddings_chembert_lmdb/`

```bash
python scripts/prepare_raw_dataset.py \
  --input_csv "/abs/path/to/your_raw.csv" \
  --dataset_name YOUR_DATASET \
  --dataset_kind ligand
```

如果是虚拟筛选数据，则把 `--dataset_kind` 改成 `vs`。

### 2) 查看训练脚本参数

```bash
python scripts/finetune_pharmaPrompt.py --help
```

### 3) 最小训练 smoke test（HPK1）

这条命令已实际进入训练计算；由于显存占用较高，使用了较小 batch 和 AMP。

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

## 测试 / 推理

### 1) 查看推理脚本参数

```bash
python scripts/predict_approved.py --help
python scripts/predict_drugbank_top20.py --help
```

### 2) HPK1 评估（evaluate）

这条命令已实际跑通；当前显存占用较高时，使用 `cuda:3` 和较小 `batch_size` 更稳。

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

### 3) EGFR 已批准药物预测

这条命令已实际跑通，并生成 `approved_predictions.csv`。

```bash
python scripts/predict_approved.py \
  --model_name EGFR \
  --approved_name EGFR_approved_2 \
  --device cuda:0 \
  --batch_size 16 \
  --num_workers 0
```

#!/usr/bin/env python3
"""
Single-SMILES pharmacophore attribution (DeepSHAP) for PharmAgent.

What it does:
- Runs the trained model on (target_smiles + background_smiles).
- Extracts the intermediate prompt tensor `prompt_feat` (shape [B, 27, D]) and the graph readout.
- Uses shap.DeepExplainer to attribute the *final prediction* to the prompt_feat groups.
- Aggregates |SHAP| into 8 pharmacophore categories via PHAR_NUM_LIST and plots a bar chart.

Notes:
- This is heavier than the simple `pred_phar_num` evidence output.
- Results depend on the chosen background distribution.
"""

import argparse
import base64
import io
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# Canonical 8 category names for output (JSON key "ZnBinder" normalized to "ZincBinder")
PHAR8_NAMES = [
    "Donor",
    "Acceptor",
    "NegIonizable",
    "PosIonizable",
    "ZincBinder",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
]
# Fallback counts per category if JSON/phar_question_name unavailable; must sum to 27
PHAR_NUM_LIST = [1, 1, 1, 4, 6, 5, 2, 7]


def _load_phar8_from_json(repo: Path):
    """Build type->category mapping and 27-question order from phar_question JSON.
    Returns (type_to_cat, phar_question_order_27). phar_question_order_27 is the
    list of type names in JSON iteration order (so index i = model prompt index i
    if the model uses the same JSON/key order). ZnBinder -> ZincBinder.
    """
    import json
    path = repo / "datasets" / "text" / "phar_question_howmany_gpt4o_27.json"
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    type_to_cat: Dict[str, str] = {}
    phar_question_order_27: List[str] = []
    for key, items in data.items():
        cat = "ZincBinder" if key == "ZnBinder" else key
        for item in items:
            t = item.get("type", "")
            if t:
                type_to_cat[t] = cat
            phar_question_order_27.append(t or "")
    return type_to_cat, phar_question_order_27


def _load_phar_questions_json(repo: Path) -> Optional[Dict[str, list]]:
    path = repo / "datasets" / "text" / "phar_question_howmany_gpt4o_27.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _canonical_phar8_name(name: str) -> str:
    return "ZincBinder" if name == "ZnBinder" else name


def _smarts_match_count(mol, smarts: str) -> int:
    from rdkit import Chem

    if mol is None or not smarts:
        return 0
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return int(len(mol.GetSubstructMatches(patt)))


def _phar8_match_counts(repo: Path, smiles: str) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    from rdkit import Chem

    data = _load_phar_questions_json(repo)
    cat_counts: Dict[str, int] = {k: 0 for k in PHAR8_NAMES}
    question_counts: Dict[str, Dict[str, int]] = {k: {} for k in PHAR8_NAMES}
    if data is None:
        return cat_counts, question_counts

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return cat_counts, question_counts

    for raw_cat, items in data.items():
        cat = _canonical_phar8_name(raw_cat)
        if cat not in cat_counts:
            continue
        for item in items:
            type_name = str(item.get("type", "") or "")
            count = _smarts_match_count(mol, str(item.get("SMARTS", "") or ""))
            if type_name:
                question_counts[cat][type_name] = count
            cat_counts[cat] += count
    return cat_counts, question_counts


def _find_repo_root() -> Path:
    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen = set()
    for start in starts:
        for p in [start] + list(start.parents):
            if p in seen:
                continue
            seen.add(p)
            if (p / "server" / "inference_service.py").exists():
                return p
    raise RuntimeError("Cannot locate repo root (missing server/inference_service.py).")


def _ensure_rdkit_data_path() -> None:
    """Repair broken RDKit wheel paths in relocated environments.

    Some packaged environments keep RDKit installed but ship an invalid
    build-time RDDataDir. Before importing modules that transitively import
    RDKit feature factories, point RDBASE at a real share/RDKit directory
    when one exists.
    """
    candidates = [
        Path(sys.executable).resolve().parent.parent / "share" / "RDKit",
        Path(sys.prefix).resolve() / "share" / "RDKit",
        Path("/root/envs/KPGT/share/RDKit"),
    ]
    for candidate in candidates:
        if (candidate / "Data" / "BaseFeatures.fdef").exists():
            os.environ["RDBASE"] = str(candidate)
            return


def _safe_output_stem(model: str, smiles: str, ts: Optional[str] = None) -> str:
    """Build a filesystem-safe output stem from model + target smiles + timestamp."""
    model_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model).strip()).strip("._-") or "model"
    raw = str(smiles).strip() or "target"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    if not safe:
        safe = "target"
    safe = safe[:80]
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_safe}_{safe}_{ts}"


def _read_smiles_from_csv(csv_path: Path, col: str) -> List[str]:
    import pandas as pd

    df = pd.read_csv(str(csv_path))
    if col == "auto":
        col = "smiles" if "smiles" in df.columns else ("SMILES" if "SMILES" in df.columns else "")
    if not col or col not in df.columns:
        raise ValueError(f"Cannot find smiles column '{col}' in {csv_path}. Available columns: {list(df.columns)}")

    vals = df[col].dropna().astype(str).tolist()
    # de-dup while preserving order
    seen = set()
    out: List[str] = []
    for s in vals:
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _default_background_csv(repo: Path, model: str) -> str:
    """
    Auto-pick a reasonable background CSV for the task.

    Priority:
    - FGFR1_IC50 / HPK1_IC50: `datasets/ligand/<task>/<task>.csv` first, then `datasets/vs/...`
    - others: `datasets/vs/<task>/<task>.csv` first, then `datasets/ligand/...`
    """
    cand: List[Path] = []
    if model in {"FGFR1_IC50", "HPK1_IC50"}:
        cand.append(repo / "datasets" / "ligand" / model / f"{model}.csv")
        cand.append(repo / "datasets" / "vs" / model / f"{model}.csv")
    elif (repo / "datasets" / "moleculeace" / model / f"{model}.csv").exists():
        cand.append(repo / "datasets" / "moleculeace" / model / f"{model}.csv")
        cand.append(repo / "datasets" / "vs" / model / f"{model}.csv")
    else:
        cand.append(repo / "datasets" / "vs" / model / f"{model}.csv")
        cand.append(repo / "datasets" / "ligand" / model / f"{model}.csv")

    for p in cand:
        if p.exists():
            return str(p)
    return ""


def _load_background_smiles(
    repo: Path,
    model: str,
    n: int,
    *,
    background_csv: str,
    background_col: str,
    background_mode: str,
    seed: int,
) -> Tuple[List[str], str]:
    """
    Returns: (background_smiles, source_desc)
    """
    bg_csv = (background_csv or "").strip()
    if not bg_csv:
        bg_csv = _default_background_csv(repo, model=model)

    vals: List[str] = []
    src = ""

    if bg_csv:
        p = Path(bg_csv).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"background_csv not found: {p}")
        vals = _read_smiles_from_csv(p, background_col)
        src = str(p)
    else:
        # fallback to DrugBank_cache if present
        default_csv = repo / "datasets" / "vs" / "DrugBank_cache" / "DrugBank_cache.csv"
        if default_csv.exists():
            vals = _read_smiles_from_csv(default_csv, "auto")
            src = str(default_csv)

    if vals:
        if background_mode == "head":
            return (vals[:n] if n > 0 else []), src
        if background_mode == "random":
            rng = random.Random(int(seed))
            if n <= 0:
                return [], src
            if len(vals) <= n:
                return vals, src
            return rng.sample(vals, n), src
        raise ValueError(f"Unknown background_mode: {background_mode}")

    # final fallback
    fallback = ["CCO", "c1ccccc1", "c1ccccc1O", "CC(=O)O", "CCN", "O=C=O", "CC(C)O", "CCOC(=O)C"]
    return fallback[:n], "(builtin_fallback)"


def _default_ckpt_path(repo: Path, model: str) -> Path:
    seed42_ace_ckpt = (
        repo
        / "save"
        / model
        / "question_num8"
        / "ace"
        / "seed_42"
        / "kpgt_False"
        / "prompt_cat"
        / "Noise_False"
        / "text_model_pubmed"
        / "base_encoder_graph"
        / "best_model.pth"
    )
    if seed42_ace_ckpt.exists():
        return seed42_ace_ckpt
    if model in {"EGFR", "JAK1", "bace", "bbbp"}:
        return (
            repo
            / "save"
            / model
            / "question_num8"
            / "scaffold-0"
            / "seed_42"
            / "text_model_pubmed"
            / "base_encoder_LiGhT"
            / "alpha_0.1_beta_0.1"
            / "best_model.pth"
        )
    if model in {"FGFR1_IC50", "HPK1_IC50"}:
        return (
            repo
            / "save"
            / model
            / "question_num8"
            / "scaffold-3"
            / "seed_42"
            / "text_model_pubmed"
            / "base_encoder_LiGhT"
            / "alpha_0.1_beta_0.1"
            / "best_model.pth"
        )
    if model in {"CHEMBL1862_Ki"}:
        return (
            repo
            / "save"
            / model
            / "question_num8"
            / "ace"
            / "seed_24"
            / "text_model_pubmed"
            / "base_encoder_LiGhT"
            / "KBAN_fusion_vk"
            / "alpha_0.1_beta_0.1"
            / "ace_clip_test_False"
            / "best_model.pth"
        )
    raise ValueError(f"Unsupported model for default ckpt: {model}")


def _default_dataset_base_path(repo: Path, model: str) -> Path:
    if (repo / "datasets" / "moleculeace" / model / f"{model}.csv").exists():
        return repo / "datasets" / "moleculeace"
    if model in {"FGFR1_IC50", "HPK1_IC50"}:
        return repo / "datasets" / "ligand"
    return repo / "datasets" / "vs"


def _default_train_split_name(model: str) -> str:
    if model in {"FGFR1_IC50", "HPK1_IC50"}:
        return "scaffold-3"
    return "scaffold-0"


def _ensure_training_artifacts(repo: Path, model: str, path_length: int) -> Tuple[Path, Path]:
    dataset_base_path = _default_dataset_base_path(repo, model)
    dataset_base_arg = os.path.relpath(dataset_base_path, repo)
    split_name = _default_train_split_name(model)
    split_path = dataset_base_path / model / "splits" / f"{split_name}.npy"
    graph_path = dataset_base_path / model / f"{model}_{path_length}.pkl"

    env = os.environ.copy()

    if not split_path.exists():
        cmd = [
            sys.executable,
            str(repo / "scripts" / "preprocess_scaffold_split.py"),
            "--root_path",
            dataset_base_arg,
            "--use_split_method",
            "random_scaffold_split",
            "--dataset",
            model,
            "--seed",
            "0",
        ]
        subprocess.run(cmd, cwd=str(repo), env=env, check=True)

    if not graph_path.exists():
        cmd = [
            sys.executable,
            str(repo / "scripts" / "preprocess_dataset_graph.py"),
            "--data_path",
            dataset_base_arg,
            "--dataset",
            model,
            "--path_length",
            str(path_length),
            "--n_jobs",
            "2",
        ]
        subprocess.run(cmd, cwd=str(repo), env=env, check=True)

    return split_path, graph_path


def _load_train_label_stats(repo: Path, model: str, path_length: int) -> Tuple[float, float]:
    from dgl.data.utils import load_graphs

    split_path, graph_path = _ensure_training_artifacts(repo, model, path_length)

    train_idx = np.load(split_path, allow_pickle=True)[0]
    _, label_dict = load_graphs(str(graph_path))
    labels = label_dict["labels"][train_idx].numpy()
    mean = float(np.squeeze(np.nanmean(labels, axis=0)))
    std = float(np.squeeze(np.nanstd(labels, axis=0)))
    return mean, std


def _build_cache_dataset(service, smiles: List[str]) -> Tuple[object, object, object, object]:
    service._ensure_repo_imports()
    torch = service._torch()
    from torch.utils.data import DataLoader

    smiles = service._filter_valid_smiles(smiles)
    if not smiles:
        raise ValueError("No valid SMILES for attribution.")

    cache_dataset_name = service._ensure_dataset_from_smiles(smiles)
    service._maybe_run_preprocess(cache_dataset_name)

    collator = service._Collator(max_length=128)
    ds = service._Dataset(
        root_path=os.path.join(service.dataset_base_path, service.cache_dir_name),
        dataset=cache_dataset_name,
        dataset_type="regression",
        path_length=service.path_length,
        split_name="scaffold-0",
        split=None,
        train_kpgt="False",
        text_max_len=128,
        phar_question_name=[],
        use_norm_reg=True,
        smi_encoder_name=service.smiles_model_name,
        phar_load_method="lmdb",
        smiles_load_method="lmdb",
        split_method="kpgt",
        base_model_encoder="LiGhT",
        ace_clip_test=False,
        prompt_graph_feature_extractor="LiGhT",
        seed=42,
    )
    loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=False, drop_last=False, collate_fn=collator)
    loaded = service._ensure_loaded(dataset_for_init=ds)
    return ds, loader, loaded, torch


def _forward_prompt_feat_and_readout(model, device, text_dict, batch, torch, output_mean=None, output_std=None):
    import dgl
    import numpy as _np
    from torch.nn.utils.rnn import pad_sequence

    (
        b_smiles,
        g,
        fp,
        md,
        _labels,
        _phar_targets,
        _phar_target_mx,
        _atom_phar_target_map,
        smiles_embed,
        smiles_mask,
    ) = batch

    g = g.to(device)
    fp = fp.to(device)
    md = md.to(device)
    smiles_embed = smiles_embed.to(device)
    smiles_mask = smiles_mask.to(device)

    try:
        g_work = g.clone()
    except Exception:
        g_work = g.local_var()

    indicators = g_work.ndata["vavn"]
    node_h = model.node_emb(g_work.ndata["begin_end"], indicators)
    edge_h = model.edge_emb(g_work.ndata["edge"], indicators)
    triplet_h = model.triplet_emb(node_h, edge_h, fp, md, indicators)
    triplet_h = model.model(g_work, triplet_h)
    g_work.ndata["ht"] = triplet_h

    fp_vn = triplet_h[indicators == 1]
    md_vn = triplet_h[indicators == 2]
    graph_other_feats = torch.stack([fp_vn, md_vn], dim=1)

    g2 = g_work
    g2.remove_nodes(_np.where(indicators.detach().cpu().numpy() >= 1)[0])

    num_nodes_per_graph = g2.batch_num_nodes()
    node_feat_seqs = torch.split(triplet_h[indicators <= 0], num_nodes_per_graph.tolist())
    node_feats = pad_sequence(node_feat_seqs, batch_first=True)
    node_mask = (
        torch.arange(node_feats.size(1)).to(node_feats.device)[None, :]
        < torch.tensor(num_nodes_per_graph, device=node_feats.device)[:, None]
    ).float()

    node_feats = torch.cat([graph_other_feats, node_feats], dim=1)
    node_mask = torch.cat([torch.ones_like(graph_other_feats[:, :, 0]), node_mask], dim=1)

    text_question_embeddings = text_dict["question_text_embeddings"]
    text_question_mask = text_dict["question_masks"]
    text_disp_embeddings = text_dict["discription_text_embeddings"]
    text_disp_mask = text_dict["discription_mask"]

    smiles_embed_reprs = model.smiles_embed_proj(smiles_embed)
    text_question_reprs = model.text_question_prompt_proj(text_question_embeddings)
    text_disp_reprs = model.text_disp_prompt_proj(text_disp_embeddings)

    logits_list = []
    batch_size = node_feats.size(0)

    batch_text_question_reprs = text_question_reprs.unsqueeze(1).repeat(1, batch_size, 1, 1)
    batch_text_disp_reprs = text_disp_reprs.unsqueeze(1).repeat(1, batch_size, 1, 1)
    batch_text_question_mask = text_question_mask.unsqueeze(1).repeat(1, batch_size, 1).float()
    batch_text_disp_mask = text_disp_mask.unsqueeze(1).repeat(1, batch_size, 1).float()

    for idx in range(len(text_question_embeddings)):
        kno_question_out = batch_text_question_reprs[idx]
        kno_question_out_mask = batch_text_question_mask[idx]

        kno_dis_out = batch_text_disp_reprs[idx]
        kno_dis_out_mask = batch_text_disp_mask[idx]

        kno_dis_out = torch.cat((smiles_embed_reprs, kno_dis_out), dim=1)
        kno_dis_out_mask = torch.cat((smiles_mask.float(), kno_dis_out_mask), dim=1)

        answer_out, _atten_vq, _atten_vk, _atten_qk = model.bcn_list(
            node_feats,
            kno_question_out,
            kno_dis_out,
            node_mask,
            kno_question_out_mask,
            kno_dis_out_mask,
            softmax=True,
        )
        logits_list.append(answer_out)

    prompt_feat = torch.stack(logits_list, dim=1)  # [B, 27, D]
    molecules_prompt = _build_prompt_representation(model, prompt_feat)

    readout = dgl.readout_nodes(g2, "ht", op=model.readout_mode)
    if not getattr(model, "legacy_ace_graph_readout_only", False):
        readout = torch.cat([fp_vn, md_vn, readout], dim=-1)

    g_feats = torch.cat([molecules_prompt, readout], dim=-1)
    pred = model.predictor(g_feats)
    if (output_mean is not None) and (output_std is not None):
        pred = pred * output_std + output_mean

    return list(b_smiles), pred, prompt_feat, readout


def _group_prompt_feat_to_phar8(prompt_feat):
    grouped = []
    start_q = 0
    for n_q in PHAR_NUM_LIST:
        end_q = start_q + n_q
        grouped.append(prompt_feat[:, start_q:end_q, :].mean(dim=1))
        start_q = end_q
    return torch.stack(grouped, dim=1)


def _build_prompt_representation(model, prompt_feat):
    if getattr(model, "legacy_seed42_ace_prompt_projection", False):
        if getattr(model, "legacy_seed42_ace_use_group_mlps", False):
            grouped_prompt = []
            start_q = 0
            for idx, n_q in enumerate(PHAR_NUM_LIST):
                end_q = start_q + n_q
                grouped_prompt.append(
                    model.pharma_projection_model_list[idx](prompt_feat[:, start_q:end_q, :].reshape(prompt_feat.shape[0], -1))
                )
                start_q = end_q
            grouped_prompt = torch.stack(grouped_prompt, dim=1)
        else:
            grouped_prompt = _group_prompt_feat_to_phar8(prompt_feat)
        return model.prompt_projection_model(grouped_prompt.reshape(grouped_prompt.shape[0], -1))
    return model.prompt_fusion(prompt_feat)


def _make_explain_model(model, prompt_D: int, readout_dim: int, torch, output_mean=None, output_std=None):
    class ExplainPrompt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt_fusion = getattr(model, "prompt_fusion", None)
            self.prompt_projection_model = getattr(model, "prompt_projection_model", None)
            self.pharma_projection_model_list = getattr(model, "pharma_projection_model_list", None)
            self.predictor = model.predictor
            self.legacy_seed42_ace_prompt_projection = getattr(model, "legacy_seed42_ace_prompt_projection", False)
            self.legacy_seed42_ace_use_group_mlps = getattr(model, "legacy_seed42_ace_use_group_mlps", False)
            self.output_mean = output_mean
            self.output_std = output_std

        def forward(self, x):
            prompt_flat = x[:, : 27 * prompt_D]
            readout = x[:, 27 * prompt_D :]
            prompt_feat = prompt_flat.view(-1, 27, prompt_D)
            if self.legacy_seed42_ace_prompt_projection:
                if self.legacy_seed42_ace_use_group_mlps:
                    grouped_prompt = []
                    start_q = 0
                    for idx, n_q in enumerate(PHAR_NUM_LIST):
                        end_q = start_q + n_q
                        grouped_prompt.append(
                            self.pharma_projection_model_list[idx](
                                prompt_feat[:, start_q:end_q, :].reshape(prompt_feat.shape[0], -1)
                            )
                        )
                        start_q = end_q
                    grouped_prompt = torch.stack(grouped_prompt, dim=1)
                else:
                    grouped_prompt = _group_prompt_feat_to_phar8(prompt_feat)
                molecules_prompt = self.prompt_projection_model(grouped_prompt.reshape(grouped_prompt.shape[0], -1))
            else:
                molecules_prompt = self.prompt_fusion(prompt_feat)
            g_feats = torch.cat([molecules_prompt, readout], dim=-1)
            pred = self.predictor(g_feats)
            if (self.output_mean is not None) and (self.output_std is not None):
                pred = pred * self.output_std + self.output_mean
            return pred

    return ExplainPrompt()


def _aggregate_phar8(
    abs_prompt: np.ndarray,
    prompt_D: int,
    phar_question_name: Optional[List[str]] = None,
    type_to_cat: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Aggregate |SHAP| over 27 prompt dims into 8 pharmacophore categories.
    If phar_question_name and type_to_cat are provided (from model + JSON),
    use them so index->category matches the model's question order; otherwise
    use fixed PHAR8_NAMES + PHAR_NUM_LIST (assumes JSON key order).
    """
    out: Dict[str, float] = {k: 0.0 for k in PHAR8_NAMES}

    if phar_question_name is not None and type_to_cat is not None and len(phar_question_name) == 27:
        for q_idx, type_name in enumerate(phar_question_name):
            cat = type_to_cat.get(type_name)
            if cat is None:
                cat = "LumpedHydrophobe"
            idx0 = q_idx * prompt_D
            idx1 = (q_idx + 1) * prompt_D
            out[cat] = out.get(cat, 0.0) + float(abs_prompt[idx0:idx1].sum())
        return out

    start_q = 0
    for name, n_q in zip(PHAR8_NAMES, PHAR_NUM_LIST):
        q0 = start_q
        q1 = start_q + n_q
        idx0 = q0 * prompt_D
        idx1 = q1 * prompt_D
        out[name] = float(abs_prompt[idx0:idx1].sum())
        start_q = q1
    return out


def _phar8_question_counts(
    phar_question_name: Optional[List[str]] = None,
    type_to_cat: Optional[Dict[str, str]] = None,
) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in PHAR8_NAMES}

    if phar_question_name is not None and type_to_cat is not None and len(phar_question_name) == 27:
        for type_name in phar_question_name:
            cat = type_to_cat.get(type_name)
            if cat is None:
                cat = "LumpedHydrophobe"
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    for name, n_q in zip(PHAR8_NAMES, PHAR_NUM_LIST):
        counts[name] = int(n_q)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--smiles", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--background_n", type=int, default=32)
    ap.add_argument("--background_csv", default="", help="CSV path to draw background SMILES from (optional).")
    ap.add_argument("--background_col", default="auto", help="SMILES column name in background CSV (auto=smiles/SMILES).")
    ap.add_argument("--background_mode", choices=["head", "random"], default="head")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_path", default="", help="Override finetuned checkpoint path (best_model.pth).")
    ap.add_argument("--out_json", default="deepshap_result.json")
    ap.add_argument("--out_png", default="deepshap_phar8.png")
    ap.add_argument("--test_aggregation_only", action="store_true", help="Only test JSON load + phar8 aggregation (no model/server).")
    args = ap.parse_args()

    if args.test_aggregation_only:
        if not args.model:
            args.model = "FGFR1_IC50"
        if not args.smiles:
            args.smiles = "c1ccccc1"
        repo = Path(__file__).resolve().parent.parent
        type_to_cat, phar_order_27 = _load_phar8_from_json(repo)
        if type_to_cat is None or phar_order_27 is None:
            print("FAIL: _load_phar8_from_json returned None (missing JSON?)")
            sys.exit(1)
        print(f"Loaded {len(phar_order_27)} question types from JSON; {len(type_to_cat)} type->cat mappings.")
        prompt_D = 8
        np.random.seed(42)
        abs_prompt = np.random.rand(27 * prompt_D).astype(np.float64)
        phar8_abs = _aggregate_phar8(abs_prompt, prompt_D, phar_question_name=phar_order_27, type_to_cat=type_to_cat)
        total = sum(phar8_abs.values()) or 1.0
        phar8_norm = {k: float(v) / float(total) for k, v in phar8_abs.items()}
        print("phar8_importance_abs:", json.dumps(phar8_abs, indent=2))
        print("phar8_importance_norm:", json.dumps(phar8_norm, indent=2))
        print("OK: aggregation test passed.")
        return

    if not args.model or not args.smiles:
        ap.error("--model and --smiles are required unless --test_aggregation_only is set.")
    repo = _find_repo_root()
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    _ensure_rdkit_data_path()

    from server.inference_service import InferenceService

    ckpt = Path(args.ckpt_path).expanduser().resolve() if args.ckpt_path else _default_ckpt_path(repo, args.model)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    bg, bg_src = _load_background_smiles(
        repo,
        model=str(args.model),
        n=int(args.background_n),
        background_csv=str(args.background_csv),
        background_col=str(args.background_col).strip() or "auto",
        background_mode=str(args.background_mode),
        seed=int(args.seed),
    )
    print(f"[BG] model={args.model} mode={args.background_mode} n={args.background_n} seed={args.seed} src={bg_src}")

    target_smi = str(args.smiles).strip()
    all_smiles = [target_smi] + bg

    svc = InferenceService(
        repo_root=str(repo),
        dataset_base_path=str(_default_dataset_base_path(repo, str(args.model))),
        pretrained_base_path=str(repo / "pretrained" / "base" / "base.pth"),
        finetuned_ckpt_path=str(ckpt),
        device_str=str(args.device),
        num_workers=2,
    )

    if not svc._filter_valid_smiles([target_smi]):
        raise ValueError(f"Invalid target SMILES: {target_smi}")

    _ds, loader, loaded, torch = _build_cache_dataset(svc, all_smiles)
    model = loaded.model
    device = loaded.device
    text_dict = loaded.text_dict

    output_mean = None
    output_std = None
    if str(args.model) == "EGFR":
        mean_value, std_value = _load_train_label_stats(repo, str(args.model), svc.path_length)
        output_mean = torch.tensor(mean_value, dtype=torch.float32, device=device)
        output_std = torch.tensor(std_value, dtype=torch.float32, device=device)

    model.eval()

    all_inputs = []
    all_preds = []
    out_smiles: List[str] = []
    prompt_D = None
    readout_dim = None

    with torch.no_grad():
        for batch in loader:
            b_smiles, pred, prompt_feat, readout = _forward_prompt_feat_and_readout(
                model,
                device,
                text_dict,
                batch,
                torch,
                output_mean=output_mean,
                output_std=output_std,
            )
            pred_np = pred.detach().cpu().numpy().reshape(-1)
            pf = prompt_feat.detach().cpu().numpy()
            ro = readout.detach().cpu().numpy()
            prompt_D = int(prompt_feat.shape[-1])
            readout_dim = int(readout.shape[-1])

            for i, smi in enumerate(b_smiles):
                out_smiles.append(str(smi))
                all_preds.append(float(pred_np[i]))
                all_inputs.append(np.concatenate([pf[i].reshape(-1), ro[i].reshape(-1)], axis=0))

    if prompt_D is None or readout_dim is None:
        raise RuntimeError("Failed to infer prompt/readout dims.")

    # locate target index
    try:
        target_idx = out_smiles.index(target_smi)
    except ValueError:
        raise RuntimeError(f"Target SMILES missing after dataset/model pipeline: {target_smi}")

    x = np.stack(all_inputs, axis=0)
    x_target = x[target_idx : target_idx + 1]
    x_bg = np.delete(x, target_idx, axis=0)
    if x_bg.shape[0] == 0:
        x_bg = x_target.copy()

    explain_model = _make_explain_model(
        model,
        prompt_D=prompt_D,
        readout_dim=readout_dim,
        torch=torch,
        output_mean=output_mean,
        output_std=output_std,
    )
    explain_model.eval()

    import shap

    bg_t = torch.tensor(x_bg, dtype=torch.float32, device=device)
    xt_t = torch.tensor(x_target, dtype=torch.float32, device=device)

    explainer = shap.DeepExplainer(explain_model, bg_t)
    shap_vals = explainer.shap_values(xt_t)
    shap_arr = np.asarray(shap_vals[0] if isinstance(shap_vals, list) else shap_vals).reshape(-1)

    abs_shap = np.abs(shap_arr)
    abs_prompt = abs_shap[: 27 * prompt_D]
    # Use type-based aggregation only when the server provides phar_question_name in model order.
    # Otherwise use fixed PHAR_NUM_LIST (Donor 0:1, Acceptor 1:2, ..., Aromatic 13:18, ...) so we
    # don't assume model order matches our local JSON iteration order.
    type_to_cat, _ = _load_phar8_from_json(repo)
    phar_question_name = None
    if isinstance(text_dict.get("phar_question_name"), list) and len(text_dict.get("phar_question_name")) == 27:
        phar_question_name = text_dict.get("phar_question_name")
    phar8_abs = _aggregate_phar8(
        abs_prompt,
        prompt_D=prompt_D,
        phar_question_name=phar_question_name,
        type_to_cat=type_to_cat if phar_question_name else None,
    )
    phar8_aggregation_mode = "phar_question_name" if phar_question_name else "fixed_PHAR_NUM_LIST"

    phar8_question_counts = _phar8_question_counts(
        phar_question_name=phar_question_name,
        type_to_cat=type_to_cat if phar_question_name else None,
    )
    phar8_match_counts, phar8_question_match_counts = _phar8_match_counts(repo, out_smiles[target_idx])
    phar8_present_mask = {k: bool(phar8_match_counts.get(k, 0) > 0) for k in PHAR8_NAMES}
    phar8_mean_abs = {
        k: float(v) / float(max(phar8_question_counts.get(k, 1), 1)) for k, v in phar8_abs.items()
    }

    sum_total = sum(phar8_abs.values()) or 1.0
    phar8_sum_norm = {k: float(v) / float(sum_total) for k, v in phar8_abs.items()}

    mean_total_all = sum(phar8_mean_abs.values()) or 1.0
    phar8_mean_norm_all = {k: float(v) / float(mean_total_all) for k, v in phar8_mean_abs.items()}

    present_items = {k: float(v) for k, v in phar8_mean_abs.items() if phar8_present_mask.get(k, False)}
    if present_items:
        mean_total_present = sum(present_items.values()) or 1.0
        phar8_mean_norm = {
            k: (float(present_items[k]) / float(mean_total_present)) if k in present_items else 0.0
            for k in PHAR8_NAMES
        }
        phar8_rank_mode = "mean_per_question_present_only"
    else:
        phar8_mean_norm = dict(phar8_mean_norm_all)
        phar8_rank_mode = "mean_per_question"

    topk = sorted(
        [(k, v) for k, v in phar8_mean_norm.items() if v > 0.0],
        key=lambda kv: kv[1],
        reverse=True,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labs = [k for k, _ in topk][::-1]
    vals = [v for _, v in topk][::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(labs))))
    ax.barh(labs, vals, color="#4C78A8")
    ax.set_xlabel("Mean Importance Per Question", fontsize=15, fontweight="bold")
    ax.set_title(f"{args.model} | pred={all_preds[target_idx]:.4f}", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    png_bytes = buf.getvalue()

    out_dir = Path.cwd() / "buffer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = _safe_output_stem(args.model, target_smi)
    out_png_path = out_dir / f"{out_stem}.png"
    out_json_path = out_dir / f"{out_stem}.json"

    out_png_path.write_bytes(png_bytes)

    out = {
        "model": args.model,
        "smiles": out_smiles[target_idx],
        "prediction": float(all_preds[target_idx]),
        "background_n": int(x_bg.shape[0]),
        "background_source": bg_src,
        "phar8_aggregation_mode": phar8_aggregation_mode,
        "phar8_rank_mode": phar8_rank_mode,
        "phar8_question_counts": phar8_question_counts,
        "phar8_match_counts": phar8_match_counts,
        "phar8_present_mask": phar8_present_mask,
        "phar8_question_match_counts": phar8_question_match_counts,
        "phar8_importance_abs": phar8_abs,
        "phar8_importance_sum_norm": phar8_sum_norm,
        "phar8_importance_mean_abs": phar8_mean_abs,
        "phar8_importance_mean_norm_all": phar8_mean_norm_all,
        "phar8_importance_mean_norm": phar8_mean_norm,
        "phar8_importance_norm": phar8_mean_norm,
        "phar8_topk_norm": [{"name": k, "score": float(v)} for k, v in topk],
        "phar8_topk_mean_norm": [{"name": k, "score": float(v)} for k, v in topk],
        "plot_png_path": str(out_png_path),
        "plot_png_base64": base64.b64encode(png_bytes).decode("ascii"),
        "ckpt_path": str(ckpt),
    }
    out_json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_json_path} and {out_png_path}")


if __name__ == "__main__":
    main()


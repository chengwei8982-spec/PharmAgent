from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LoadedArtifacts:
    model: Any
    device: Any
    text_dict: dict
    phar_question_name: List[str]


class InferenceService:
    """
    A pragmatic server-side inference wrapper for this repo.

    Design choice:
    - Reuse existing dataset/preprocess pipeline to minimize refactoring risk.
    - Cache per-request dataset artifacts under datasets/vs/<cache_dir>/<hash>/...

    This is optimized for batch inference and reproducible runs, not for ultra-low latency.
    """

    def __init__(
        self,
        repo_root: str,
        dataset_base_path: str,
        pretrained_base_path: str,
        finetuned_ckpt_path: str,
        device_str: str = "cuda:0",
        cache_dir_name: str = "_server_cache",
        path_length: int = 5,
        num_workers: int = 8,
        text_model_name: str = "pubmed",
        smiles_model_name: str = "chembert",
        train_text_model: bool = False,
        dropout: float = 0.0,
        n_layers: int = 2,
        projection_layers: int = 2,
    ):
        self.repo_root = os.path.abspath(repo_root)
        self.dataset_base_path = os.path.abspath(dataset_base_path)
        self.pretrained_base_path = os.path.abspath(pretrained_base_path)
        self.finetuned_ckpt_path = os.path.abspath(finetuned_ckpt_path)

        self.device_str = device_str
        self.cache_dir_name = cache_dir_name
        self.path_length = int(path_length)
        self.num_workers = int(num_workers)

        self.text_model_name = text_model_name
        self.smiles_model_name = smiles_model_name
        self.train_text_model = bool(train_text_model)
        self.dropout = float(dropout)
        self.n_layers = int(n_layers)
        self.projection_layers = int(projection_layers)

        self._lock = threading.Lock()
        self._loaded: Optional[LoadedArtifacts] = None

        # Repo imports are delayed until first prediction call (so the app can import without torch/rdkit).
        self._repo_imported = False

    def _ensure_repo_imports(self) -> None:
        if self._repo_imported:
            return
        sys.path.append(self.repo_root)
        from src.data.collator import Collator_pharmagent  # noqa
        from src.data.finetune_dataset import PharmaQADataset  # noqa
        from src.model.light import TextEncoder  # noqa
        from src.model_config import config_dict  # noqa

        self._Collator = Collator_pharmagent
        self._Dataset = PharmaQADataset
        self._TextEncoder = TextEncoder
        self._config_dict = config_dict
        self._repo_imported = True

    def _torch(self):
        import torch

        return torch

    def _device(self):
        torch = self._torch()
        if torch.cuda.is_available() and self.device_str.startswith("cuda"):
            return torch.device(self.device_str)
        return torch.device("cpu")

    def _hash_smiles(self, smiles: List[str]) -> str:
        norm = "\n".join([s.strip() for s in smiles]).encode("utf-8")
        return hashlib.sha256(norm).hexdigest()[:16]

    def _filter_valid_smiles(self, smiles: List[str]) -> List[str]:
        from rdkit import Chem

        out: List[str] = []
        for smi in smiles:
            smi = str(smi).strip()
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            out.append(smi)
        return out

    def _ensure_dataset_from_smiles(self, smiles: List[str]) -> str:
        """
        Create cache dataset dir:
          datasets/vs/<cache_dir_name>/<cache_name>/<cache_name>.csv
        """
        cache_root = os.path.join(self.dataset_base_path, self.cache_dir_name)
        os.makedirs(cache_root, exist_ok=True)

        cache_name = f"req_{self._hash_smiles(smiles)}"
        out_dir = os.path.join(cache_root, cache_name)
        os.makedirs(out_dir, exist_ok=True)

        model_csv_path = os.path.join(out_dir, f"{cache_name}.csv")
        if not os.path.exists(model_csv_path):
            df = pd.DataFrame({"smiles": smiles, "label": np.zeros(len(smiles), dtype=float)})
            df.to_csv(model_csv_path, index=False)

        return cache_name

    def _maybe_run_preprocess(self, cache_dataset_name: str):
        ds_dir = os.path.join(self.dataset_base_path, self.cache_dir_name, cache_dataset_name)
        need = {
            "graph_pkl": os.path.join(ds_dir, f"{cache_dataset_name}_{self.path_length}.pkl"),
            "fp": os.path.join(ds_dir, "rdkfp1-7_512.npz"),
            "md": os.path.join(ds_dir, "molecular_descriptors.npz"),
            "phar_lmdb": os.path.join(ds_dir, "phar_features_lmdb"),
            "smi_lmdb": os.path.join(ds_dir, "smiles_embeddings_chembert_lmdb"),
        }
        missing = []
        for k, p in need.items():
            if k.endswith("_lmdb"):
                if not os.path.isdir(p):
                    missing.append(k)
            else:
                if not os.path.exists(p):
                    missing.append(k)
        if not missing:
            return

        # graph/fp/md
        subprocess.check_call(
            [
                sys.executable,
                os.path.join(self.repo_root, "scripts", "preprocess_dataset_graph.py"),
                "--data_path",
                os.path.join(self.dataset_base_path, self.cache_dir_name),
                "--dataset",
                cache_dataset_name,
                "--path_length",
                str(self.path_length),
                "--n_jobs",
                str(self.num_workers),
            ],
            cwd=self.repo_root,
        )

        # pharmacophore features
        subprocess.check_call(
            [
                sys.executable,
                os.path.join(self.repo_root, "scripts", "preprocess_dataset_phar.py"),
                "--dataset_base_path",
                os.path.join(self.dataset_base_path, self.cache_dir_name),
                "--dataset",
                cache_dataset_name,
                "--input",
                "phar",
                "--multiprocessing",
                "False",
                "--num_workers",
                str(max(1, min(8, self.num_workers))),
            ],
            cwd=self.repo_root,
        )

        # smiles embedding（chembert）
        subprocess.check_call(
            [
                sys.executable,
                os.path.join(self.repo_root, "scripts", "preprocess_dataset_phar.py"),
                "--dataset_base_path",
                os.path.join(self.dataset_base_path, self.cache_dir_name),
                "--dataset",
                cache_dataset_name,
                "--input",
                "smiles_embedding",
                "--multiprocessing",
                "False",
                "--num_workers",
                str(max(1, min(8, self.num_workers))),
            ],
            cwd=self.repo_root,
        )

    def _build_text_dict(self, device: Any) -> Tuple[dict, Any, List[str]]:
        torch = self._torch()
        text_path = os.path.join(os.path.dirname(self.dataset_base_path), "text", "phar_question_howmany_gpt4o_27.json")
        with open(text_path, "r", encoding="utf-8") as fp:
            text_list = __import__("json").load(fp)

        select_question = []
        select_description = []
        phar_question_name: List[str] = []
        for _, items in text_list.items():
            for item in items:
                select_question.append(f"Question: {item['question']}")
                select_description.append(f"Description: {item['description']}")
                phar_question_name.append(item["type"])

        text_model = self._TextEncoder(model_name=self.text_model_name, load=True)
        q_ids, q_masks = text_model.tokenize(select_question, max_length=96)
        d_ids, d_masks = text_model.tokenize(select_description, max_length=96)

        if q_ids.dim() == 1:
            q_ids = q_ids.unsqueeze(0)
            q_masks = q_masks.unsqueeze(0)
            d_ids = d_ids.unsqueeze(0)
            d_masks = d_masks.unsqueeze(0)

        if not self.train_text_model:
            with torch.no_grad():
                q_emb = text_model(input_ids=q_ids, attention_mask=q_masks)
                d_emb = text_model(input_ids=d_ids, attention_mask=d_masks)
            text_dict = {
                "question_text_embeddings": q_emb.to(device),
                "discription_text_embeddings": d_emb.to(device),
                "question_masks": q_masks.to(device),
                "discription_mask": d_masks.to(device),
                "phar_question_name": phar_question_name,
                "question_tokens": q_ids,
                "know_tokens": d_ids,
                "question_texts": select_question,
            }
        else:
            text_dict = {
                "question_masks": q_masks.to(device),
                "discription_mask": d_masks.to(device),
                "phar_question_name": phar_question_name,
                "question_tokens": q_ids.to(device),
                "know_tokens": d_ids.to(device),
                "question_texts": select_question,
            }
        return text_dict, text_model, phar_question_name

    def _init_model_for_inference(self, device: Any, dataset) -> Any:
        # Reuse logic from scripts/predict_drugbank_top20.py to avoid drifting.
        from scripts.predict_drugbank_top20 import init_model_for_inference  # type: ignore

        model = init_model_for_inference(
            device=device,
            pretrained_base_path=self.pretrained_base_path,
            finetuned_ckpt_path=self.finetuned_ckpt_path,
            dataset=dataset,
            text_model_name=self.text_model_name,
            smiles_model_name=self.smiles_model_name,
            dropout=self.dropout,
            n_layers=self.n_layers,
            projection_layers=self.projection_layers,
            train_text_model=self.train_text_model,
            text_model=None,
        )
        return model

    def _ensure_loaded(self, dataset_for_init) -> LoadedArtifacts:
        self._ensure_repo_imports()
        with self._lock:
            if self._loaded is not None:
                return self._loaded

            device = self._device()
            text_dict, _text_model, phar_question_name = self._build_text_dict(device=device)
            model = self._init_model_for_inference(device=device, dataset=dataset_for_init)
            self._loaded = LoadedArtifacts(model=model, device=device, text_dict=text_dict, phar_question_name=phar_question_name)
            return self._loaded

    def predict_smiles(self, smiles: List[str], return_attention: bool = False) -> Tuple[List[str], np.ndarray, Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        self._ensure_repo_imports()
        torch = self._torch()
        from torch.utils.data import DataLoader

        if not os.path.exists(self.pretrained_base_path):
            raise FileNotFoundError(f"Missing pretrained base weights: {self.pretrained_base_path}")
        if not os.path.exists(self.finetuned_ckpt_path):
            raise FileNotFoundError(f"Missing finetuned checkpoint: {self.finetuned_ckpt_path}")

        smiles = self._filter_valid_smiles(smiles)
        if not smiles:
            raise ValueError("No valid SMILES provided.")

        cache_dataset_name = self._ensure_dataset_from_smiles(smiles)
        self._maybe_run_preprocess(cache_dataset_name)

        # Load dataset + collator
        collator = self._Collator(max_length=128)
        ds = self._Dataset(
            root_path=os.path.join(self.dataset_base_path, self.cache_dir_name),
            dataset=cache_dataset_name,
            dataset_type="regression",
            path_length=self.path_length,
            split_name="scaffold-0",
            split=None,
            train_kpgt="False",
            text_max_len=128,
            phar_question_name=[],
            use_norm_reg=False,
            smi_encoder_name=self.smiles_model_name,
            phar_load_method="lmdb",
            smiles_load_method="lmdb",
            split_method="kpgt",
            base_model_encoder="LiGhT",
            ace_clip_test=False,
            prompt_graph_feature_extractor="LiGhT",
            seed=42,
        )
        loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=False, drop_last=False, collate_fn=collator)

        loaded = self._ensure_loaded(dataset_for_init=ds)
        model = loaded.model
        device = loaded.device
        text_dict = loaded.text_dict

        all_smiles: List[str] = []
        all_pred: List[float] = []
        all_phar: List[np.ndarray] = []
        all_att: List[np.ndarray] = []

        model.eval()
        with torch.no_grad():
            for batch in loader:
                (
                    b_smiles,
                    graphs,
                    fps,
                    mds,
                    labels,
                    phar_targets,
                    phar_target_mx,
                    atom_phar_target_map,
                    smiles_embed,
                    smiles_mask,
                ) = batch

                graphs = graphs.to(device)
                fps = fps.to(device)
                mds = mds.to(device)
                smiles_embed = smiles_embed.to(device)
                smiles_mask = smiles_mask.to(device)

                pred, pred_phar_num, atten = model.forward_pharmagent(
                    graphs, fps, mds, text=text_dict, smiles_embed=smiles_embed, smiles_mask=smiles_mask
                )

                pred = pred.detach().cpu().numpy().reshape(-1)
                phar = pred_phar_num.detach().cpu().numpy()

                all_smiles.extend(list(b_smiles))
                all_pred.extend(pred.tolist())
                all_phar.append(phar)
                if return_attention:
                    all_att.append(atten.detach().cpu().numpy())

        phar_arr = np.concatenate(all_phar, axis=0) if all_phar else None
        att_arr = np.concatenate(all_att, axis=0) if (return_attention and all_att) else None

        return all_smiles, np.asarray(all_pred, dtype=float), phar_arr, att_arr, loaded.phar_question_name


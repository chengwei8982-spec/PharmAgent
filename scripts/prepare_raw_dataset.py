import argparse
import json
import os
import pickle
import subprocess
import sys

import lmdb
import pandas as pd
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from tqdm import tqdm


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from src.model.light import TextEncoder


def _resolve_base_features_path():
    candidates = []

    rddata_dir = getattr(RDConfig, "RDDataDir", None)
    if rddata_dir:
        candidates.append(os.path.join(rddata_dir, "BaseFeatures.fdef"))

    candidates.extend(
        [
            os.path.join(sys.prefix, "share", "RDKit", "Data", "BaseFeatures.fdef"),
            os.path.join(
                sys.prefix,
                "lib",
                "python%s.%s" % sys.version_info[:2],
                "site-packages",
                "rdkit",
                "Data",
                "BaseFeatures.fdef",
            ),
        ]
    )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    raise OSError("BaseFeatures.fdef could not be found. Checked: %s" % ", ".join(candidates))


FDEF_NAME = _resolve_base_features_path()
FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(FDEF_NAME)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a raw CSV dataset into PharmAgent trainable format."
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Path to raw CSV file.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Output dataset folder name.")
    parser.add_argument(
        "--dataset_kind",
        type=str,
        required=True,
        choices=["ligand", "vs"],
        help="Dataset family. ligand is used for training/evaluation; vs is used for virtual screening.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Override output root. Defaults to datasets/<dataset_kind> under repo root.",
    )
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--graph_n_jobs", type=int, default=8)
    parser.add_argument("--phar_num_workers", type=int, default=8)
    parser.add_argument(
        "--split_seeds",
        type=str,
        default="",
        help="Comma-separated seeds for scaffold splits. Defaults: ligand=3,4,5 ; vs=0",
    )
    return parser.parse_args()


def normalize_csv(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    if "smiles" in df.columns:
        smiles_col = "smiles"
    elif "SMILES" in df.columns:
        smiles_col = "SMILES"
    elif "Smiles" in df.columns:
        smiles_col = "Smiles"
    else:
        raise ValueError("Input CSV must contain one of: smiles, SMILES, Smiles")

    df = df.dropna(how="all").copy()
    df = df.dropna(subset=[smiles_col]).copy()
    df = df.rename(columns={smiles_col: "smiles"})
    df.to_csv(output_csv, index=False)


def run_cmd(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def load_smiles_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError(f"Normalized CSV must contain 'smiles': {csv_path}")
    return df["smiles"].astype(str).tolist()


def load_phar_question_names():
    text_path = os.path.join(REPO_ROOT, "datasets", "text", "phar_question_howmany_gpt4o_27.json")
    with open(text_path, "r", encoding="utf-8") as fp:
        text_list = json.load(fp)

    phar_question_name = []
    for _, items in text_list.items():
        for item in items:
            phar_question_name.append(item["type"])
    return phar_question_name


def extract_phar_feature(smiles: str, phar_question_name):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    feats = FEATURE_FACTORY.GetFeaturesForMol(mol)
    num_nodes = mol.GetNumAtoms()

    attn_phar_atom = torch.zeros(len(phar_question_name), len(feats))
    for phar_index, phar_task in enumerate(phar_question_name):
        matching_phars = [feat.GetType() == phar_task for feat in feats]
        attn_phar_atom[phar_index, matching_phars] = 1

    phar_targets_num = attn_phar_atom.sum(dim=1).long()
    atom_phar_target_map = torch.zeros(num_nodes, len(phar_question_name))
    phar_target_mx = torch.zeros((len(mol.GetBonds()), len(phar_question_name)))
    fp_name = []

    for feat in feats:
        ids = feat.GetAtomIds()
        fp_name.append([".".join((feat.GetFamily(), feat.GetType())), ids, list(feat.GetPos())])
        phar_index = phar_question_name.index(feat.GetType())
        atom_phar_target_map[ids, phar_index] = 1

    bonded_atoms = set()
    for bond_index, bond in enumerate(mol.GetBonds()):
        begin_atom_id, end_atom_id = sorted([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
        for question_idx in range(len(phar_question_name)):
            phar_target_mx[bond_index, question_idx] = atom_phar_target_map[
                [begin_atom_id, end_atom_id], question_idx
            ].sum(dim=0).to(torch.bool).to(torch.float32)
        bonded_atoms.add(begin_atom_id)
        bonded_atoms.add(end_atom_id)

    atom_feat_list = []
    for atom_id in range(num_nodes):
        if atom_id not in bonded_atoms:
            atom_phar = torch.zeros(len(phar_question_name))
            for question_idx in range(len(phar_question_name)):
                atom_phar[question_idx] = atom_phar_target_map[[atom_id], question_idx].sum(dim=0).to(
                    torch.bool
                ).to(torch.float32)
            atom_feat_list.append(atom_phar)

    if atom_feat_list:
        atom_feat_tensor = torch.stack(atom_feat_list, dim=0)
        phar_target_mx = torch.cat((phar_target_mx, atom_feat_tensor), dim=0)

    return {
        "fp_name": fp_name,
        "phar_targets_num": phar_targets_num,
        "atom_phar_target_map": atom_phar_target_map,
        "phar_target_mx": phar_target_mx,
    }


def extract_smiles_embedding_chembert(smiles: str, text_model):
    idx_list, idx_mask, adj_mask, adj_matx = text_model.tokenizer.encode(smiles)
    smiles_embeddings = text_model.model(
        input=torch.tensor(idx_list, dtype=torch.long).unsqueeze(0),
        imask=torch.tensor(idx_mask, dtype=torch.bool).unsqueeze(0),
        amask=torch.tensor(adj_mask, dtype=torch.float).unsqueeze(0),
        amatx=torch.tensor(adj_matx, dtype=torch.float).unsqueeze(0),
    ).squeeze()

    return {
        "smiles_embeddings": smiles_embeddings,
        "smiles_mask": torch.tensor(idx_mask, dtype=torch.long),
    }


def save_lmdb(records, out_path: str):
    os.makedirs(out_path, exist_ok=True)
    env = lmdb.open(out_path, map_size=100 * 1024 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, record in enumerate(records):
            txn.put(str(i).encode(), pickle.dumps(record))
    env.close()


def build_feature_lmdbs(dataset_dir: str, csv_path: str):
    smiles_list = load_smiles_from_csv(csv_path)
    phar_question_name = load_phar_question_names()

    print("[RUN] building pharmacophore LMDB")
    phar_records = [extract_phar_feature(smiles, phar_question_name) for smiles in tqdm(smiles_list)]
    save_lmdb(phar_records, os.path.join(dataset_dir, "phar_features_lmdb"))

    print("[RUN] building ChemBERT embedding LMDB")
    text_model = TextEncoder(model_name="chembert", load=True)
    smiles_records = [extract_smiles_embedding_chembert(smiles, text_model) for smiles in tqdm(smiles_list)]
    save_lmdb(smiles_records, os.path.join(dataset_dir, "smiles_embeddings_chembert_lmdb"))


def main():
    args = parse_args()

    if args.output_root.strip():
        output_root = args.output_root.strip()
        if os.path.basename(os.path.normpath(output_root)) != args.dataset_kind:
            output_root = os.path.join(output_root, args.dataset_kind)
    else:
        output_root = os.path.join(REPO_ROOT, "datasets", args.dataset_kind)
    dataset_dir = os.path.join(output_root, args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    output_csv = os.path.join(dataset_dir, f"{args.dataset_name}.csv")
    normalize_csv(args.input_csv, output_csv)
    print(f"[OK] normalized csv -> {output_csv}")

    if args.split_seeds.strip():
        split_seeds = [s.strip() for s in args.split_seeds.split(",") if s.strip()]
    else:
        split_seeds = ["3", "4", "5"] if args.dataset_kind == "ligand" else ["0"]

    for seed in split_seeds:
        run_cmd(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "scripts", "preprocess_scaffold_split.py"),
                "--root_path",
                os.path.relpath(output_root, REPO_ROOT),
                "--use_split_method",
                "random_scaffold_split",
                "--dataset",
                args.dataset_name,
                "--seed",
                seed,
            ]
        )

    run_cmd(
        [
            sys.executable,
            os.path.join(REPO_ROOT, "scripts", "preprocess_dataset_graph.py"),
            "--data_path",
            output_root,
            "--dataset",
            args.dataset_name,
            "--path_length",
            str(args.path_length),
            "--n_jobs",
            str(args.graph_n_jobs),
        ]
    )

    build_feature_lmdbs(dataset_dir=dataset_dir, csv_path=output_csv)

    print("[DONE] Dataset is ready for training/inference.")
    print(f"[DONE] Output folder: {dataset_dir}")


if __name__ == "__main__":
    main()

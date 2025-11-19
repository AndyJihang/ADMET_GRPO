import os
import pandas as pd
from typing import List, Dict


def _load_caco2_wang(path: str) -> List[Dict]:
    df = pd.read_csv(path, sep="\t")

    data = []
    for _, row in df.iterrows():
        smiles = str(row["Drug"])
        y = float(row["Y"])
        drug_id = str(row.get("Drug_ID", ""))

        problem = (
            "You are an ADMET prediction assistant.\n\n"
            "Task:\n"
            "Given a molecule represented by SMILES, predict its Caco-2 permeability "
            "(Wang dataset, unit: log(cm/s)).\n\n"
            "Requirements:\n"
            "- Return ONLY a single float number (no units, no explanation).\n"
            "- Use reasonable scientific prior, but do not hallucinate impossible values.\n\n"
            f"Molecule SMILES: {smiles}\n"
            "Answer:"
        )

        sample = {
            "problem": problem,
            "groundtruth": y,
            "smiles": smiles,
            "drug_id": drug_id,
        }
        data.append(sample)

    return data


def load_data(name: str):
    """
    Entry point used by train.py
    """
    if name == "caco2_wang":
        path = os.getenv("CACO2_WANG_PATH", None)
        if path is None:
            this_dir = os.path.dirname(__file__)
            path = os.path.join(this_dir, "data", "caco2_wang.tab")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"caco2_wang file not found at: {path}. "
                "Please set CACO2_WANG_PATH or put file at admet/data/caco2_wang.tab."
            )

        return _load_caco2_wang(path)

    else:
        raise ValueError(f"Unsupported dataset: {name}")
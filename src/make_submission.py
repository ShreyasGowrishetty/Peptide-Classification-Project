import os
import sys
import numpy as np
import pickle
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from features import sequence_features, pdb_features, TRIMERS

LABEL_COLS = [
    'antibacterial', 'anticancer', 'antifungal', 'antihypertensive',
    'antimicrobial', 'antiparasitic', 'antiviral',
    'cell_cell_communication', 'drug_delivery_vehicle', 'toxic'
]

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

def sequence_from_pdb(pdb_path):
    """Extract amino acid sequence from ATOM records in a PDB file."""
    seen_residues = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                res_name  = line[17:20].strip()
                try:
                    res_num = int(line[22:26].strip())
                except ValueError:
                    continue
                if atom_name == "CA" and res_name in AA3TO1:
                    seen_residues[res_num] = AA3TO1[res_name]
    return ''.join(seen_residues[k] for k in sorted(seen_residues))


def load_model(use_structure=False):
    suffix = "struct" if use_structure else "seq"
    with open(f"models/model_{suffix}.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(f"models/scaler_{suffix}.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(f"models/thresholds_{suffix}.pkl", 'rb') as f:
        thresholds = pickle.load(f)
    with open(f"models/trimer_idx_{suffix}.pkl", 'rb') as f:
        trimer_idx = pickle.load(f)
    return model, scaler, thresholds, trimer_idx


def build_feature_vector(seq, pdb_path, use_structure, trimer_idx):
    """Build the same feature vector used during training."""
    base, dimers, trimers = sequence_features(seq)
    struct = pdb_features(pdb_path if use_structure else None)
    trimers_top = trimers[trimer_idx]
    if use_structure:
        return np.concatenate([base, dimers, trimers_top, struct])
    else:
        return np.concatenate([base, dimers, trimers_top])


def make_submission(test_pdb_dir, output_csv, use_structure=True):
    print(f"Loading model (structure={use_structure})...")
    model, scaler, thresholds, trimer_idx = load_model(use_structure)

    pdb_files = sorted([f for f in os.listdir(test_pdb_dir) if f.endswith('.pdb')])
    print(f"Found {len(pdb_files)} test PDB files")

    ids, X_list = [], []
    for fname in pdb_files:
        pid      = fname.replace('.pdb', '')
        pdb_path = os.path.join(test_pdb_dir, fname)
        seq      = sequence_from_pdb(pdb_path)

        if len(seq) == 0:
            print(f"  WARNING: empty sequence for {fname}, skipping")
            continue

        feats = build_feature_vector(seq, pdb_path, use_structure, trimer_idx)
        ids.append(pid)
        X_list.append(feats)

    print(f"Extracted features for {len(ids)} peptides")

    X        = np.array(X_list, dtype=np.float32)
    X_scaled = scaler.transform(X)

    probs = np.column_stack([
        est.predict_proba(X_scaled)[:, 1]
        for est in model.estimators_
    ])

    out = pd.DataFrame({'ID': ids})
    for i, col in enumerate(LABEL_COLS):
        out[col] = probs[:, i]

    out.to_csv(output_csv, index=False)
    print(f"\nSubmission saved to {output_csv}")
    print(f"\nSample:")
    print(out.head(5).to_string())
    return out


if __name__ == "__main__":
    make_submission(
        test_pdb_dir="data/kaggle/test/pdb",
        output_csv="submission_struct.csv",
        use_structure=True
    )
    make_submission(
        test_pdb_dir="data/kaggle/test/pdb",
        output_csv="submission_seq.csv",
        use_structure=False
    )
    print("\nDone! Upload submission_struct.csv and submission_seq.csv to Kaggle.")
import numpy as np
import pickle
import os
import sys
import sqlite3
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from features import load_dataset, extract_all_features

LABEL_COLS = [
    'antibacterial', 'anticancer', 'antifungal', 'antihypertensive',
    'antimicrobial', 'antiparasitic', 'antiviral',
    'cell_cell_communication', 'drug_delivery_vehicle', 'toxic'
]

def load_model(use_structure=False):
    """Load saved model, scaler, and thresholds."""
    suffix = "struct" if use_structure else "seq"
    with open(f"models/model_{suffix}.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(f"models/scaler_{suffix}.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(f"models/thresholds_{suffix}.pkl", 'rb') as f:
        thresholds = pickle.load(f)
    return model, scaler, thresholds


def predict_from_csv(input_csv, output_csv, use_structure=False, pdb_dir=None):
    """
    Read a CSV of peptides and write predictions to output_csv.
    Input CSV must have columns: ID, sequence
    Optionally: pdb_dir for structure mode
    """
    print(f"Loading model (structure={use_structure})...")
    model, scaler, thresholds = load_model(use_structure)

    print(f"Reading input: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"Extracting features for {len(df)} peptides...")
    X_list = []
    for _, row in df.iterrows():
        pid = row['ID']
        seq = row['sequence']
        pdb_path = None
        if use_structure and pdb_dir:
            pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")
        feats = extract_all_features(seq, pdb_path)
        X_list.append(feats)

    X = np.array(X_list, dtype=np.float32)
    X_scaled = scaler.transform(X)

    # Get probabilities
    probs = np.column_stack([
        est.predict_proba(X_scaled)[:, 1]
        for est in model.estimators_
    ])

    # Apply per-class thresholds
    preds = (probs >= thresholds).astype(int)

    # Build output dataframe
    # Format: ID, then one column per label (probability)
    # Kaggle wants probabilities so they can apply their own threshold
    out = pd.DataFrame({'ID': df['ID']})
    for i, col in enumerate(LABEL_COLS):
        out[col] = probs[:, i]

    out.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    print(f"\nSample output:")
    print(out.head(3).to_string())
    return out


def predict_from_db(db_path, output_csv, use_structure=False, pdb_dir=None):
    """
    Run predictions on the training DB (useful for sanity checking).
    """
    print(f"Loading model (structure={use_structure})...")
    model, scaler, thresholds = load_model(use_structure)

    X, Y, ids, label_cols = load_dataset(
        db_path=db_path,
        pdb_dir=pdb_dir,
        use_structure=use_structure
    )

    X_scaled = scaler.transform(X)
    probs = np.column_stack([
        est.predict_proba(X_scaled)[:, 1]
        for est in model.estimators_
    ])

    out = pd.DataFrame({'ID': ids})
    for i, col in enumerate(LABEL_COLS):
        out[col] = probs[:, i]

    out.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    print(f"\nSample output:")
    print(out.head(3).to_string())
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run peptide activity predictions")
    parser.add_argument("--input",     required=True,  help="Input CSV with ID and sequence columns")
    parser.add_argument("--output",    required=True,  help="Output CSV path for predictions")
    parser.add_argument("--structure", action="store_true", help="Use structure mode (requires PDB files)")
    parser.add_argument("--pdb_dir",   default="data/pdb", help="Directory containing PDB files")
    args = parser.parse_args()

    predict_from_csv(
        input_csv=args.input,
        output_csv=args.output,
        use_structure=args.structure,
        pdb_dir=args.pdb_dir
    )
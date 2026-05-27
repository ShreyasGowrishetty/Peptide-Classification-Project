import numpy as np
import sqlite3
import os
from itertools import product

# The 20 standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Hydrophobicity scale (Kyte-Doolittle)
HYDROPHOBICITY = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}

# Charge at physiological pH
CHARGE = {
    'A': 0,  'C': 0,  'D': -1, 'E': -1, 'F': 0,
    'G': 0,  'H': 0,  'I': 0,  'K': 1,  'L': 0,
    'M': 0,  'N': 0,  'P': 0,  'Q': 0,  'R': 1,
    'S': 0,  'T': 0,  'V': 0,  'W': 0,  'Y': 0
}

# Precompute all 2-mers and 3-mers (done once at import time)
DIMERS  = [''.join(p) for p in product(AMINO_ACIDS, repeat=2)]   # 400 combinations
TRIMERS = [''.join(p) for p in product(AMINO_ACIDS, repeat=3)]   # 8000 combinations
DIMER_INDEX  = {d: i for i, d in enumerate(DIMERS)}
TRIMER_INDEX = {t: i for i, t in enumerate(TRIMERS)}


def sequence_features(seq):
    """
    Convert a peptide sequence into a rich numeric feature vector.
    Includes amino acid composition, physicochemical properties,
    and k-mer frequencies (digrams and trigrams).
    """
    seq = seq.upper().strip()
    length = len(seq)

    # --- 1. Amino acid composition (20 features) ---
    composition = [seq.count(aa) / length for aa in AMINO_ACIDS]

    # --- 2. Physicochemical features (10 features) ---
    hydro  = [HYDROPHOBICITY.get(aa, 0) for aa in seq]
    charge = [CHARGE.get(aa, 0) for aa in seq]

    mean_hydro    = np.mean(hydro)
    std_hydro     = np.std(hydro)
    max_hydro     = np.max(hydro)
    min_hydro     = np.min(hydro)
    net_charge    = sum(charge)
    pos_fraction  = sum(1 for v in charge if v > 0) / length
    neg_fraction  = sum(1 for v in charge if v < 0) / length
    norm_length   = length / 100.0
    aromatic_frac = sum(1 for aa in seq if aa in "FWY") / length
    # Amphipathicity proxy: std of hydrophobicity along sequence
    half = length // 2
    amphipathicity = abs(np.mean(hydro[:half]) - np.mean(hydro[half:])) if length > 2 else 0

    physicochemical = [
        mean_hydro, std_hydro, max_hydro, min_hydro,
        net_charge, pos_fraction, neg_fraction,
        norm_length, aromatic_frac, amphipathicity
    ]

    # --- 3. Dimer frequencies (400 features) ---
    # Count every pair of consecutive amino acids
    dimer_counts = np.zeros(400, dtype=np.float32)
    for i in range(length - 1):
        d = seq[i:i+2]
        if d in DIMER_INDEX:
            dimer_counts[DIMER_INDEX[d]] += 1
    if length > 1:
        dimer_counts /= (length - 1)

    # --- 4. Trimer frequencies (top 200 only to keep size manageable) ---
    # 8000 trimers is too many — use only the 200 most common ones
    # We'll select them dynamically in load_dataset
    trimer_counts = np.zeros(len(TRIMERS), dtype=np.float32)
    for i in range(length - 2):
        t = seq[i:i+3]
        if t in TRIMER_INDEX:
            trimer_counts[TRIMER_INDEX[t]] += 1
    if length > 2:
        trimer_counts /= (length - 2)

    return (
        np.array(composition + physicochemical, dtype=np.float32),
        dimer_counts,
        trimer_counts
    )


def pdb_features(pdb_path):
    """Extract simple structural features from a PDB file."""
    zeros = np.zeros(4, dtype=np.float32)
    if pdb_path is None or not os.path.exists(pdb_path):
        return zeros
    try:
        ca_coords = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append([x, y, z])
        if len(ca_coords) < 2:
            return zeros
        ca_coords = np.array(ca_coords)
        end_to_end = np.linalg.norm(ca_coords[-1] - ca_coords[0])
        centroid   = ca_coords.mean(axis=0)
        rog        = np.sqrt(np.mean(np.sum((ca_coords - centroid)**2, axis=1)))
        diffs      = ca_coords[:, None, :] - ca_coords[None, :, :]
        max_dist   = np.sqrt((diffs**2).sum(axis=2)).max()
        n_residues = len(ca_coords) / 100.0
        return np.array([end_to_end, rog, max_dist, n_residues], dtype=np.float32)
    except Exception:
        return zeros

def extract_all_features(seq, pdb_path=None):
    """
    Full feature vector for one peptide.
    Calls sequence_features and pdb_features and concatenates them.
    Used by predict.py and make_submission.py for single peptide prediction.
    """
    base, dimers, trimers = sequence_features(seq)
    struct = pdb_features(pdb_path)

    # We can't apply the trimer variance filter here (no dataset context)
    # so we just use all trimers — the scaler handles normalization
    # In practice the saved scaler was fit on the top-200 trimers,
    # so we need to be consistent. We'll handle this in make_submission directly.
    return np.concatenate([base, dimers, struct])

def load_dataset(db_path, pdb_dir=None, use_structure=False, top_trimers=200):
    """
    Load dataset and return feature matrix X, label matrix Y, ids, label_cols.
    Uses amino acid composition + physicochemical + dimers + top trimers.
    """
    label_cols = [
        'antibacterial', 'anticancer', 'antifungal', 'antihypertensive',
        'antimicrobial', 'antiparasitic', 'antiviral',
        'cell_cell_communication', 'drug_delivery_vehicle', 'toxic'
    ]

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM peptides").fetchall()
    col_names = [d[0] for d in conn.execute("SELECT * FROM peptides LIMIT 1").description]
    conn.close()

    ids = []
    base_list, dimer_list, trimer_list, struct_list, Y_list = [], [], [], [], []

    for row in rows:
        r   = dict(zip(col_names, row))
        pid = r['ID']
        seq = r['sequence']

        pdb_path = None
        if use_structure and pdb_dir:
            pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")

        base, dimers, trimers = sequence_features(seq)
        struct = pdb_features(pdb_path)
        labels = [r[c] for c in label_cols]

        ids.append(pid)
        base_list.append(base)
        dimer_list.append(dimers)
        trimer_list.append(trimers)
        struct_list.append(struct)
        Y_list.append(labels)

    base_arr   = np.array(base_list,   dtype=np.float32)
    dimer_arr  = np.array(dimer_list,  dtype=np.float32)
    trimer_arr = np.array(trimer_list, dtype=np.float32)
    struct_arr = np.array(struct_list, dtype=np.float32)
    Y          = np.array(Y_list,      dtype=np.int32)

    # Select top trimers by variance — most variable = most informative
    trimer_var     = trimer_arr.var(axis=0)
    top_idx        = np.argsort(trimer_var)[-top_trimers:]
    trimer_arr_top = trimer_arr[:, top_idx]

    if use_structure:
        X = np.concatenate([base_arr, dimer_arr, trimer_arr_top, struct_arr], axis=1)
    else:
        X = np.concatenate([base_arr, dimer_arr, trimer_arr_top], axis=1)

    print(f"Feature breakdown: {base_arr.shape[1]} base + {dimer_arr.shape[1]} dimers "
          f"+ {trimer_arr_top.shape[1]} trimers"
          + (f" + {struct_arr.shape[1]} structural" if use_structure else ""))
    print(f"Total features: {X.shape[1]}")

    return X, Y, ids, label_cols, top_idx

if __name__ == "__main__":
    X, Y, ids, label_cols = load_dataset(
        db_path="data/labels.sqlite",
        pdb_dir="data/pdb",
        use_structure=True
    )
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label matrix shape:   {Y.shape}")
    print(f"First peptide ID: {ids[0]}")
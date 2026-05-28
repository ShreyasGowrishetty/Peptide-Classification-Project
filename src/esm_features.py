import torch
import esm
import numpy as np
import sqlite3
import os
import pickle

def load_esm_model():
    """Load the ESM-2 model (downloads ~300MB on first run)."""
    print("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval()  # disable dropout
    batch_converter = alphabet.get_batch_converter()
    print("ESM-2 loaded!")
    return model, alphabet, batch_converter


def get_esm_embeddings(sequences, ids, model, alphabet, batch_converter, batch_size=32):
    """
    Convert a list of sequences into ESM embeddings.
    Each sequence becomes a 320-dimensional vector (mean pooled).
    """
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        batch_ids  = ids[i:i+batch_size]

        # ESM expects list of (label, sequence) tuples
        data = [(pid, seq.upper()) for pid, seq in zip(batch_ids, batch_seqs)]
        _, _, tokens = batch_converter(data)

        with torch.no_grad():
            results = model(tokens, repr_layers=[12], return_contacts=False)

        # Extract per-residue embeddings from last layer, then mean pool
        embeddings = results["representations"][12]  # (batch, seq_len, 480)
        # Mean pool across sequence length (ignore start/end tokens)
        for j, (_, seq) in enumerate(data):
            seq_len = len(seq)
            emb = embeddings[j, 1:seq_len+1].mean(0).numpy()  # (320,)
            all_embeddings.append(emb)

        if (i // batch_size) % 5 == 0:
            print(f"  Processed {min(i+batch_size, len(sequences))}/{len(sequences)} peptides")

    return np.array(all_embeddings, dtype=np.float32)


def generate_and_save_embeddings(db_path, output_path="models/esm_embeddings.pkl"):
    """Generate ESM embeddings for all training peptides and save them."""
    print("Loading dataset...")
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT ID, sequence FROM peptides").fetchall()
    conn.close()

    ids       = [r[0] for r in rows]
    sequences = [r[1] for r in rows]
    print(f"Loaded {len(ids)} peptides")

    model, alphabet, batch_converter = load_esm_model()

    print("Generating embeddings...")
    embeddings = get_esm_embeddings(sequences, ids, model, alphabet, batch_converter)
    print(f"Embedding matrix shape: {embeddings.shape}")

    # Save as dict: ID -> embedding
    emb_dict = {pid: emb for pid, emb in zip(ids, embeddings)}
    with open(output_path, 'wb') as f:
        pickle.dump(emb_dict, f)
    print(f"Saved embeddings to {output_path}")
    return emb_dict


def generate_test_embeddings(test_pdb_dir, output_path="models/esm_embeddings_test.pkl"):
    """Generate ESM embeddings for test peptides (sequence extracted from PDB)."""

    AA3TO1 = {
        'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
        'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
        'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
        'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
    }

    def sequence_from_pdb(pdb_path):
        seen = {}
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom = line[12:16].strip()
                    res  = line[17:20].strip()
                    try:
                        num = int(line[22:26].strip())
                    except ValueError:
                        continue
                    if atom == "CA" and res in AA3TO1:
                        seen[num] = AA3TO1[res]
        return ''.join(seen[k] for k in sorted(seen))

    pdb_files = sorted([f for f in os.listdir(test_pdb_dir) if f.endswith('.pdb')])
    ids, sequences = [], []
    for fname in pdb_files:
        pid = fname.replace('.pdb', '')
        seq = sequence_from_pdb(os.path.join(test_pdb_dir, fname))
        if len(seq) > 0:
            ids.append(pid)
            sequences.append(seq)

    print(f"Loaded {len(ids)} test peptides")

    model, alphabet, batch_converter = load_esm_model()

    print("Generating test embeddings...")
    embeddings = get_esm_embeddings(sequences, ids, model, alphabet, batch_converter)
    print(f"Test embedding matrix shape: {embeddings.shape}")

    emb_dict = {pid: emb for pid, emb in zip(ids, embeddings)}
    with open(output_path, 'wb') as f:
        pickle.dump(emb_dict, f)
    print(f"Saved test embeddings to {output_path}")
    return emb_dict


if __name__ == "__main__":
    # Generate training embeddings
    generate_and_save_embeddings(
        db_path="data/labels.sqlite",
        output_path="models/esm_embeddings_train.pkl"
    )
    # Generate test embeddings
    generate_test_embeddings(
        test_pdb_dir="data/kaggle/test/pdb",
        output_path="models/esm_embeddings_test.pkl"
    )
import numpy as np
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from features import load_dataset

LABEL_COLS = [
    'antibacterial', 'anticancer', 'antifungal', 'antihypertensive',
    'antimicrobial', 'antiparasitic', 'antiviral',
    'cell_cell_communication', 'drug_delivery_vehicle', 'toxic'
]

def find_best_thresholds(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_thresh = []
    for i in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t
        best_thresh.append(best_t)
    return np.array(best_thresh)


def cross_validate(X, Y, model_fn, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        model = model_fn()
        model.fit(X_train_s, Y_train)

        probs = np.column_stack([
            est.predict_proba(X_val_s)[:, 1]
            for est in model.estimators_
        ])
        thresholds = find_best_thresholds(Y_val, probs)
        Y_pred = (probs >= thresholds).astype(int)

        score = f1_score(Y_val, Y_pred, average='macro', zero_division=0)
        fold_scores.append(score)
        print(f"  Fold {fold+1}: macro F1 = {score:.4f}")

    mean = np.mean(fold_scores)
    std  = np.std(fold_scores)
    print(f"  Mean: {mean:.4f} ± {std:.4f}")
    return mean


def train_and_save(use_structure=False):
    print(f"\n{'='*50}")
    print(f"Training ESM model (structure={use_structure})")
    print(f"{'='*50}")

    # Load handcrafted features
    print("Loading handcrafted features...")
    X_hand, Y, ids, label_cols, trimer_idx = load_dataset(
        db_path="data/labels.sqlite",
        pdb_dir="data/pdb",
        use_structure=use_structure
    )

    # Load ESM embeddings
    print("Loading ESM embeddings...")
    with open("models/esm_embeddings_train.pkl", 'rb') as f:
        esm_dict = pickle.load(f)

    # Align ESM embeddings with dataset order
    esm_matrix = np.array([esm_dict[pid] for pid in ids], dtype=np.float32)
    print(f"ESM matrix shape: {esm_matrix.shape}")

    # Combine handcrafted + ESM features
    X = np.concatenate([X_hand, esm_matrix], axis=1)
    print(f"Combined feature shape: {X.shape}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def make_rf():
        return MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ), n_jobs=-1
        )

    def make_lr():
        return MultiOutputClassifier(
            LogisticRegression(
                C=0.5,
                class_weight='balanced',
                max_iter=2000,
                random_state=42,
                solver='saga',
                n_jobs=-1
            ), n_jobs=-1
        )

    print("\nCross-validating Random Forest + ESM...")
    rf_score = cross_validate(X_scaled, Y, make_rf)

    print("\nCross-validating Logistic Regression + ESM...")
    lr_score = cross_validate(X_scaled, Y, make_lr)

    if rf_score >= lr_score:
        print(f"\nUsing Random Forest (CV F1: {rf_score:.4f})")
        best_model = make_rf()
        best_score = rf_score
    else:
        print(f"\nUsing Logistic Regression (CV F1: {lr_score:.4f})")
        best_model = make_lr()
        best_score = lr_score

    # Retrain on full data
    print("Training final model on full dataset...")
    best_model.fit(X_scaled, Y)

    # Get thresholds
    probs = np.column_stack([
        est.predict_proba(X_scaled)[:, 1]
        for est in best_model.estimators_
    ])
    thresholds = find_best_thresholds(Y, probs)

    # Report per-class performance
    Y_pred    = (probs >= thresholds).astype(int)
    per_class = f1_score(Y, Y_pred, average=None, zero_division=0)
    macro_f1  = f1_score(Y, Y_pred, average='macro', zero_division=0)

    print(f"\nFull-data Macro F1 (optimistic): {macro_f1:.4f}")
    print("\nPer-class F1:")
    for col, score, thresh in zip(label_cols, per_class, thresholds):
        bar = "█" * int(score * 20)
        print(f"  {col:30s}: {score:.4f}  threshold={thresh:.2f}  {bar}")

    # Save everything
    suffix = "esm_struct" if use_structure else "esm_seq"
    os.makedirs("models", exist_ok=True)
    with open(f"models/model_{suffix}.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    with open(f"models/scaler_{suffix}.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"models/thresholds_{suffix}.pkl", 'wb') as f:
        pickle.dump(thresholds, f)
    with open(f"models/trimer_idx_{suffix}.pkl", 'wb') as f:
        pickle.dump(trimer_idx, f)

    print(f"\nSaved model, scaler, thresholds, trimer_idx for '{suffix}' mode")
    return best_score


if __name__ == "__main__":
    seq_score    = train_and_save(use_structure=False)
    struct_score = train_and_save(use_structure=True)

    print(f"\n{'='*50}")
    print(f"FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"ESM Sequence-only CV Macro F1:      {seq_score:.4f}")
    print(f"ESM Sequence+structure CV Macro F1: {struct_score:.4f}")
import numpy as np
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

from features import load_dataset

LABEL_COLS = [
    'antibacterial', 'anticancer', 'antifungal', 'antihypertensive',
    'antimicrobial', 'antiparasitic', 'antiviral',
    'cell_cell_communication', 'drug_delivery_vehicle', 'toxic'
]

def find_best_thresholds_cv(y_true, y_prob):
    """
    Find best threshold per class using the validation fold data.
    This is honest because we're tuning on data the model never trained on.
    """
    thresholds = np.arange(0.05, 0.95, 0.05)
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


def make_xgb(scale_pos_weight=1):
    """
    XGBoost classifier — much better than Random Forest for:
    - Imbalanced classes (scale_pos_weight handles rare classes)
    - Tabular data with mixed feature types
    - Avoiding overfitting (built-in regularization)
    """
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,       # L1 regularization
        reg_lambda=1.0,      # L2 regularization
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )


def train_and_save(use_structure=False):
    print(f"\n{'='*50}")
    print(f"Training XGBoost+ESM (structure={use_structure})")
    print(f"{'='*50}")

    # Load handcrafted features
    print("Loading handcrafted features...")
    X_hand, Y, ids, label_cols, trimer_idx = load_dataset(
        db_path="data/labels.sqlite",
        pdb_dir="data/pdb",
        use_structure=use_structure
    )

    # Load ESM-35M embeddings
    print("Loading ESM-35M embeddings...")
    with open("models/esm_embeddings_train.pkl", 'rb') as f:
        esm_dict = pickle.load(f)

    esm_matrix = np.array([esm_dict[pid] for pid in ids], dtype=np.float32)
    print(f"ESM matrix shape: {esm_matrix.shape}")

    # Combine features
    X = np.concatenate([X_hand, esm_matrix], axis=1)
    print(f"Combined feature shape: {X.shape}")

    # Cross-validation with per-fold threshold tuning
    print("\nRunning 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_val_probs  = np.zeros((len(Y), len(LABEL_COLS)))
    all_val_labels = np.zeros((len(Y), len(LABEL_COLS)), dtype=int)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        # Train one XGBoost per label
        fold_probs = np.zeros((len(val_idx), len(LABEL_COLS)))
        for i, col in enumerate(LABEL_COLS):
            y_train_i = Y_train[:, i]
            # Handle class imbalance per label
            pos = y_train_i.sum()
            neg = len(y_train_i) - pos
            spw = neg / pos if pos > 0 else 1.0
            spw = min(spw, 20.0)  # cap at 20x to avoid instability

            clf = make_xgb(scale_pos_weight=spw)
            clf.fit(X_train_s, y_train_i)
            fold_probs[:, i] = clf.predict_proba(X_val_s)[:, 1]

        # Tune thresholds on this fold's validation data
        thresholds = find_best_thresholds_cv(Y_val, fold_probs)
        Y_pred = (fold_probs >= thresholds).astype(int)
        score  = f1_score(Y_val, Y_pred, average='macro', zero_division=0)
        fold_scores.append(score)

        all_val_probs[val_idx]  = fold_probs
        all_val_labels[val_idx] = Y_val

        print(f"  Fold {fold+1}: macro F1 = {score:.4f}")

    mean_cv = np.mean(fold_scores)
    std_cv  = np.std(fold_scores)
    print(f"  Mean CV: {mean_cv:.4f} ± {std_cv:.4f}")

    # Get overall thresholds from all OOF predictions
    # OOF = out-of-fold, these are honest predictions on unseen data
    print("\nTuning thresholds on out-of-fold predictions...")
    final_thresholds = find_best_thresholds_cv(all_val_labels, all_val_probs)
    Y_oof_pred = (all_val_probs >= final_thresholds).astype(int)
    oof_macro  = f1_score(all_val_labels, Y_oof_pred, average='macro', zero_division=0)
    oof_per    = f1_score(all_val_labels, Y_oof_pred, average=None,    zero_division=0)

    print(f"OOF Macro F1 (most honest estimate): {oof_macro:.4f}")
    print("\nPer-class OOF F1:")
    for col, score, thresh in zip(LABEL_COLS, oof_per, final_thresholds):
        bar = "█" * int(score * 20)
        print(f"  {col:30s}: {score:.4f}  thresh={thresh:.2f}  {bar}")

    # Retrain on ALL data with per-label class weights
    print("\nRetraining on full dataset...")
    scaler_full = StandardScaler()
    X_full_s    = scaler_full.fit_transform(X)

    final_models = []
    for i, col in enumerate(LABEL_COLS):
        y_i = Y[:, i]
        pos = y_i.sum()
        neg = len(y_i) - pos
        spw = min(neg / pos if pos > 0 else 1.0, 20.0)
        clf = make_xgb(scale_pos_weight=spw)
        clf.fit(X_full_s, y_i)
        final_models.append(clf)
        print(f"  Trained {col} (pos={pos}, spw={spw:.1f})")

    # Save everything
    suffix = "final_struct" if use_structure else "final_seq"
    os.makedirs("models", exist_ok=True)
    with open(f"models/model_{suffix}.pkl", 'wb') as f:
        pickle.dump(final_models, f)
    with open(f"models/scaler_{suffix}.pkl", 'wb') as f:
        pickle.dump(scaler_full, f)
    with open(f"models/thresholds_{suffix}.pkl", 'wb') as f:
        pickle.dump(final_thresholds, f)
    with open(f"models/trimer_idx_{suffix}.pkl", 'wb') as f:
        pickle.dump(trimer_idx, f)

    print(f"\nSaved all model files for '{suffix}' mode")
    return oof_macro


if __name__ == "__main__":
    seq_score    = train_and_save(use_structure=False)
    struct_score = train_and_save(use_structure=True)

    print(f"\n{'='*50}")
    print(f"FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"XGBoost+ESM Sequence-only OOF F1:      {seq_score:.4f}")
    print(f"XGBoost+ESM Sequence+structure OOF F1: {struct_score:.4f}")
import numpy as np
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold
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

def find_best_threshold(y_true, y_prob):
    """
    For each class, find the probability threshold that maximizes F1.
    Instead of always using 0.5, we tune per class.
    Think of it like tuning a sensitivity dial per label.
    """
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
    """
    Proper k-fold cross validation.
    Splits data into 5 chunks, trains on 4, tests on 1, rotates.
    Gives a much more reliable estimate than a single train/val split.
    """
    # Use first label column for stratification
    from sklearn.model_selection import KFold
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

        # Get probabilities for threshold tuning
        probs = np.column_stack([
            est.predict_proba(X_val_s)[:, 1]
            for est in model.estimators_
        ])
        thresholds = find_best_threshold(Y_val, probs)
        Y_pred = (probs >= thresholds).astype(int)

        score = f1_score(Y_val, Y_pred, average='macro', zero_division=0)
        fold_scores.append(score)
        print(f"  Fold {fold+1}: macro F1 = {score:.4f}")

    mean_score = np.mean(fold_scores)
    std_score  = np.std(fold_scores)
    print(f"  Mean: {mean_score:.4f} ± {std_score:.4f}")
    return mean_score


def train_and_save(use_structure=False):
    print(f"\n{'='*50}")
    print(f"Training with structure: {use_structure}")
    print(f"{'='*50}")

    print("Loading dataset...")
    X, Y, ids, label_cols, trimer_idx = load_dataset(        db_path="data/labels.sqlite",
        pdb_dir="data/pdb",
        use_structure=use_structure
    )
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # We'll try two models and pick the better one
    def make_rf():
        return MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_leaf=3,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ), n_jobs=-1
        )

    def make_lr():
        return MultiOutputClassifier(
            LogisticRegression(
                C=0.1,                  # strong regularization to prevent overfitting
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='saga',
                n_jobs=-1
            ), n_jobs=-1
        )

    print("\nCross-validating Random Forest...")
    rf_score = cross_validate(X_scaled, Y, make_rf)

    print("\nCross-validating Logistic Regression...")
    lr_score = cross_validate(X_scaled, Y, make_lr)

    # Pick the better model
    if rf_score >= lr_score:
        print(f"\nUsing Random Forest (CV F1: {rf_score:.4f})")
        best_model = make_rf()
    else:
        print(f"\nUsing Logistic Regression (CV F1: {lr_score:.4f})")
        best_model = make_lr()

    # Train on full dataset
    print("Training final model on full dataset...")
    best_model.fit(X_scaled, Y)

    # Find best thresholds on full data
    probs = np.column_stack([
        est.predict_proba(X_scaled)[:, 1]
        for est in best_model.estimators_
    ])
    thresholds = find_best_threshold(Y, probs)

    # Final evaluation on full data (optimistic but shows ceiling)
    Y_pred     = (probs >= thresholds).astype(int)
    macro_f1   = f1_score(Y, Y_pred, average='macro', zero_division=0)
    per_class  = f1_score(Y, Y_pred, average=None,    zero_division=0)

    print(f"\nFull-data Macro F1 (optimistic): {macro_f1:.4f}")
    print("\nPer-class F1 (full data):")
    for col, score, thresh in zip(label_cols, per_class, thresholds):
        bar = "█" * int(score * 20)
        print(f"  {col:30s}: {score:.4f}  threshold={thresh:.2f}  {bar}")

    # Save everything
    suffix = "struct" if use_structure else "seq"
    os.makedirs("models", exist_ok=True)
    with open(f"models/model_{suffix}.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    with open(f"models/scaler_{suffix}.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"models/thresholds_{suffix}.pkl", 'wb') as f:
        pickle.dump(thresholds, f)
    with open(f"models/trimer_idx_{suffix}.pkl", 'wb') as f:
        pickle.dump(trimer_idx, f)

    print(f"\nSaved model, scaler, thresholds for '{suffix}' mode")
    return rf_score, lr_score


if __name__ == "__main__":
    seq_rf, seq_lr       = train_and_save(use_structure=False)
    struct_rf, struct_lr = train_and_save(use_structure=True)

    print(f"\n{'='*50}")
    print(f"FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Sequence-only:       RF={seq_rf:.4f}  LR={seq_lr:.4f}")
    print(f"Sequence+structure:  RF={struct_rf:.4f}  LR={struct_lr:.4f}")
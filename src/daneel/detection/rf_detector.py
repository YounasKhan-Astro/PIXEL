# assignment2 task g
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def create_balanced_dataset(X, y, samples_per_class=350):
    """Same idea as in the notebook: balance class 0 and 1 by augmentation / resampling."""
    X0 = X[y == 0]
    X1 = X[y == 1]

    def augment_to_target(X_orig, n_target):
        if len(X_orig) >= n_target:
            idx = np.random.choice(len(X_orig), n_target, replace=False)
            return X_orig[idx]

        X_result = [X_orig]
        while len(np.vstack(X_result)) < n_target:
            n_needed = n_target - len(np.vstack(X_result))
            idx = np.random.choice(len(X_orig), min(len(X_orig), n_needed))
            aug_type = np.random.rand()

            if aug_type < 0.25:
                X_aug = X_orig[idx] + np.random.normal(0, 0.01, (len(idx), X_orig.shape[1]))
            elif aug_type < 0.5:
                scale = 1.0 + np.random.uniform(-0.03, 0.03, (len(idx), 1))
                X_aug = X_orig[idx] * scale
            elif aug_type < 0.75:
                shifts = np.random.randint(-20, 20, len(idx))
                X_aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
            else:
                X_aug = X_orig[idx] * (1.0 + np.random.uniform(-0.02, 0.02, (len(idx), 1)))
                X_aug += np.random.normal(0, 0.008, X_aug.shape)

            X_result.append(X_aug)

        X_final = np.vstack(X_result)
        return X_final[:n_target]

    X0_bal = augment_to_target(X0, samples_per_class)
    X1_bal = augment_to_target(X1, samples_per_class)

    X_bal = np.vstack([X0_bal, X1_bal])
    y_bal = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])

    idx = np.arange(len(X_bal))
    np.random.shuffle(idx)

    return X_bal[idx], y_bal[idx]


def load_data(csv_path, n_bins=1000, use_scaler=False, samples_per_class=350):
    df = pd.read_csv(csv_path)

    flux_cols = [f"flux_{i:04d}" for i in range(n_bins)]
    flux_err_cols = [f"flux_err_{i:04d}" for i in range(n_bins)]

    X = df[flux_cols].values
    X_err = df[flux_err_cols].values
    y = df["label"].values

    metadata_cols = ["toi_name", "tic", "label", "disp", "period_d", "t0_bjd", "dur_hr", "sector"]
    metadata = df[metadata_cols]

    X_train, X_test, y_train, y_test, X_err_train, X_err_test, idx_train, idx_test = train_test_split(
        X, y, X_err, np.arange(len(y)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # balance train
    X_train, y_train = create_balanced_dataset(X_train, y_train, samples_per_class=samples_per_class)

    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    metadata_test = metadata.iloc[idx_test].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, metadata_test, scaler


def build_random_forest(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    random_state=RANDOM_STATE,
    n_jobs=-1,
):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return rf


def evaluate_with_optimal_threshold(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]

    y_pred_best = (proba >= best_thresh).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc_best = accuracy_score(y_test, y_pred_best)

    print(f"Optimal threshold: {best_thresh:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Accuracy @optimal: {acc_best:.4f} ({acc_best*100:.2f}%)")

    print("\nClassification report (optimal threshold):")
    print(classification_report(y_test, y_pred_best, target_names=["Non-Planet", "Planet"], digits=4, zero_division=0))

    cm = confusion_matrix(y_test, y_pred_best)
    precision = precision_score(y_test, y_pred_best, zero_division=0)

    return cm, precision, best_thresh


class RandomForestTransitDetector:
    """
    Simple RF-based exoplanet detector, mirroring the notebook behaviour.
    """

    def __init__(
        self,
        csv_path,
        n_bins=1000,
        use_scaler=False,
        samples_per_class=350,
        n_estimators=500,
        max_depth=None,
    ):
        self.csv_path = csv_path
        self.n_bins = n_bins
        self.use_scaler = use_scaler
        self.samples_per_class = samples_per_class
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def run(self):
        print("=" * 70)
        print("RANDOM FOREST DETECTION FROM CLI")
        print("=" * 70)

        X_train, X_test, y_train, y_test, metadata_test, scaler = load_data(
            csv_path=self.csv_path,
            n_bins=self.n_bins,
            use_scaler=self.use_scaler,
            samples_per_class=self.samples_per_class,
        )

        rf = build_random_forest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
        )

        rf.fit(X_train, y_train)
        print("Model trained on balanced training set.")

        cm, precision, best_thresh = evaluate_with_optimal_threshold(rf, X_test, y_test)

        print("\nConfusion matrix (rows: true [0,1], cols: pred [0,1]):")
        print(cm)
        print(f"Precision (optimal threshold): {precision:.4f}")

        # This mirrors what you printed into the Task D report
        print("\nDone.")

        return cm, precision, best_thresh

def run_rf_from_yaml(params_yaml):
    """
    Wrapper called by the CLI: reads parameters.yaml, instantiates the class,
    runs detection, and prints stats.
    """
    with open(params_yaml, "r") as f:
        config = yaml.safe_load(f)

    det_cfg = config.get("detection", {})

    # kernel name for bookkeeping (which RF configuration)
    kernel = det_cfg.get("kernel", "default")
    print(f"Using RF kernel: {kernel}")

    csv_path = det_cfg.get("csv_path", "tess_data.csv")
    n_bins = det_cfg.get("n_bins", 1000)
    use_scaler = det_cfg.get("use_scaler", False)
    samples_per_class = det_cfg.get("samples_per_class", 350)
    n_estimators = det_cfg.get("n_estimators", 500)
    max_depth = det_cfg.get("max_depth", None)

    detector = RandomForestTransitDetector(
        csv_path=csv_path,
        n_bins=n_bins,
        use_scaler=use_scaler,
        samples_per_class=samples_per_class,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    detector.run()

# assignment2 task h

import numpy as np
import yaml

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
)

# Re-use the same data loading + balancing as the RF detector
from daneel.detection.rf_detector import load_data

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def evaluate_with_optimal_threshold(y_test, proba):
    """Same idea as in the notebook: find best threshold and print stats."""
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
    print(
        classification_report(
            y_test,
            y_pred_best,
            target_names=["Non-Planet", "Planet"],
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred_best)
    precision = precision_score(y_test, y_pred_best, zero_division=0)

    return cm, precision, best_thresh


class CNNTransitDetector:
    """
    Simple 1D CNN-based exoplanet detector, mirroring the deep learning notebook.
    """

    def __init__(
        self,
        csv_path,
        n_bins=1000,
        use_scaler=True,
        samples_per_class=350,
        epochs=20,
        batch_size=64,
        learning_rate=1e-3,
    ):
        self.csv_path = csv_path
        self.n_bins = n_bins
        self.use_scaler = use_scaler
        self.samples_per_class = samples_per_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def build_model(self):
        """Define a small 1D CNN similar to the notebook."""
        print("Building 1D CNN model...")
        model = keras.Sequential(
            [
                layers.Input(shape=(self.n_bins, 1)),

                # Feature extraction
                layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
                layers.MaxPooling1D(2),
                layers.Dropout(0.3),

                layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
                layers.MaxPooling1D(2),
                layers.Dropout(0.3),

                layers.Conv1D(256, kernel_size=3, padding="same", activation="relu"),
                layers.MaxPooling1D(2),
                layers.Dropout(0.3),

                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        model.summary()
        return model

    def run(self):
        print("=" * 70)
        print("CNN DETECTION FROM CLI")
        print("=" * 70)

        # 1) Load + balance data (same as RF)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            metadata_test,
            scaler,
        ) = load_data(
            csv_path=self.csv_path,
            n_bins=self.n_bins,
            use_scaler=self.use_scaler,
            samples_per_class=self.samples_per_class,
        )

        # 2) Prepare for CNN: add channel dimension
        X_train_cnn = X_train.astype("float32")[..., np.newaxis]
        X_test_cnn = X_test.astype("float32")[..., np.newaxis]

        # 3) Build CNN model
        model = self.build_model()

        # 4) Train
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ]

        history = model.fit(
            X_train_cnn,
            y_train,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=2,
        )

        # 5) Predict on test and evaluate
        proba_test = model.predict(X_test_cnn).ravel()
        cm, precision, best_thresh = evaluate_with_optimal_threshold(y_test, proba_test)

        print("\nConfusion matrix (rows: true [0,1], cols: pred [0,1]):")
        print(cm)
        print(f"Precision (optimal threshold): {precision:.4f}")
        print("\nDone.")

        # You could also save the model here if you like
        # model.save("best_model_cli_cnn.keras")

        return cm, precision, best_thresh


def run_cnn_from_yaml(params_yaml):
    """
    Wrapper called by the CLI: reads parameters.yaml, instantiates CNNTransitDetector,
    runs detection, and prints statistics.
    """
    with open(params_yaml, "r") as f:
        config = yaml.safe_load(f)

    det_cfg = config.get("detection", {})

    csv_path = det_cfg.get("csv_path", "tess_data.csv")
    n_bins = det_cfg.get("n_bins", 1000)
    use_scaler = det_cfg.get("use_scaler", True)
    samples_per_class = det_cfg.get("samples_per_class", 350)
    epochs = det_cfg.get("epochs", 20)
    batch_size = det_cfg.get("batch_size", 64)
    learning_rate = det_cfg.get("learning_rate", 1e-3)

    detector = CNNTransitDetector(
        csv_path=csv_path,
        n_bins=n_bins,
        use_scaler=use_scaler,
        samples_per_class=samples_per_class,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    detector.run()

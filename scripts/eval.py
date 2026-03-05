# * - evaluate a logged mlflow run by experiment name - *
# usage: python -m scripts.eval --exp exp03_waves_rr_gboost

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

from arrhythmia_ml import file_utils
from arrhythmia_ml import ml_utils


def main(exp_name: str):
    config = file_utils.load_config()

    run_name    = config["experiments"][exp_name]["run_name"]

    ml_utils.setup_mlflow(config)

    # fetch run and load model + test data from mlflow artifacts
    run     = ml_utils.get_run_by_name(config["mlflow_experiment_name"], run_name)
    run_id  = run.info.run_id
    print(f"Loading run: {run_name}  (id: {run_id})")

    clf                      = ml_utils.load_model(run_id)
    X_test, y_test, y_train  = ml_utils.load_test_data(run_id)

    y_pred = clf.predict(X_test)

    # metrics
    class_labels, counts = np.unique(y_train, return_counts=True)
    perc     = (counts / counts.sum()) * 100
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(classification_report(y_test, y_pred))
    print(f"Macro F1: {macro_f1:.3f}")

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=class_labels,
        display_labels=class_labels,
        normalize="true",
        ax=axes[0],
    )
    axes[0].set_title(f"{run_name} | Macro F1 = {macro_f1:.3f}")

    ax   = axes[1]
    bars = ax.bar(class_labels, perc, width=0.4, color="#E65C00")

    ax.set_title("Class Distribution (Training Set)", fontsize=12)
    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, perc.max() * 1.15)

    for bar, p in zip(bars, perc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.tick_params(axis="x", labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment name from config.yaml")
    args = parser.parse_args()
    main(args.exp)

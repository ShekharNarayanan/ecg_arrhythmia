# * - train a single experiment by name - *
# usage: python -m scripts.train --exp exp03_waves_rr_gboost
# run `mlflow ui` in project root to view results

import argparse
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report

from arrhythmia_ml import file_utils
from arrhythmia_ml import ml_utils


def main(exp_name: str):
    config = file_utils.load_config()

    # resolve experiment block
    exp_cfg           = config["experiments"][exp_name]
    run_name          = exp_cfg["run_name"]
    compute_only      = exp_cfg["features"]
    classifier_name   = exp_cfg["classifier"]
    classifier_params = exp_cfg.get("classifier_params", {})
    description       = exp_cfg.get("description", "")

    # signal / feature params
    raw_data_path             = config["raw_data_path"]
    bandpass_window           = config["general_bandpass"]
    wave_extraction_window    = config["wfm_extraction_window"]
    local_rr_mean_beat_window = config["local_rr_mean_beat_window"]
    keep_labels               = np.array(config["keep_labels"])

    ml_utils.setup_mlflow(config)

    participant_ids = file_utils.get_participant_ids(raw_data_path=raw_data_path)

    # build feature matrix across all participants
    X, y, groups = ml_utils.build_feature_matrix(
        raw_data_path=raw_data_path,
        participant_ids=participant_ids,
        bandpass_window=bandpass_window,
        wave_extraction_window=wave_extraction_window,
        local_rr_mean_beat_window=local_rr_mean_beat_window,
        compute_only=compute_only,
        keep_labels=keep_labels,
    )

    X_train, X_test, y_train, y_test = ml_utils.patient_wise_split(X, y, groups)

    clf = ml_utils.build_pipeline(
        classifier_name=classifier_name,
        classifier_params=classifier_params,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("description", description)
        mlflow.log_params({
            "features"             : str(compute_only),
            "classifier"           : classifier_name,
            "bandpass"             : bandpass_window,
            "wfm_window"           : wave_extraction_window,
            "local_rr_beat_window" : local_rr_mean_beat_window,
            "n_train"              : len(X_train),
            "n_test"               : len(X_test),
        })

        print("Starting training...")
        clf = ml_utils.fit_pipeline(clf, X_train, y_train, classifier_name)
        print("Finished training.")

        y_pred = clf.predict(X_test)

        macro_f1 = ml_utils.log_metrics(y_test, y_pred, keep_labels)
        print(classification_report(y_test, y_pred))
        print(f"Macro F1: {macro_f1:.3f}")

        mlflow.sklearn.log_model(clf, artifact_path="model")
        ml_utils.save_test_data(X_test, y_test, y_train)

        print(f"Run '{run_name}' logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment name from config.yaml")
    args = parser.parse_args()
    main(args.exp)

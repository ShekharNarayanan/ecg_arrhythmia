# * - train a single experiment by name - *
# usage: python -m scripts.train --exp exp03_waves_rr_gboost
# run `mlflow ui` in project root to view results

import argparse
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from arrhythmia_ml import file_utils
from arrhythmia_ml import ml_utils
import yaml
from pathlib import Path


def main(exp_name: str):
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r") as file:
        config = yaml.safe_load(file)
        

    # resolve experiment block
    exp_cfg           = config["experiments"][exp_name]
    run_name          = exp_cfg["run_name"]
    compute_only      = exp_cfg["features"]
    classifier_name   = exp_cfg["classifier"]
    oversample_bool   = exp_cfg["oversample"]
    classifier_params = exp_cfg.get("classifier_params", {})
    search_CV_params  = exp_cfg.get("search_CV_params", {})
    description       = exp_cfg.get("description", "")

    # signal / feature params
    raw_data_path             = config["raw_data_path"]
    bandpass_window           = config["general_bandpass"]
    wave_extraction_window    = config["wfm_extraction_window"]
    local_rr_mean_beat_window = config["local_rr_mean_beat_window"]
    qrs_extraction_window     = config["qrs_extraction_window"]
    keep_labels               = np.array(config["keep_labels"])

    # model/ training params
    cv_split_param    =  config["cv_split_param"]
    n_iter_random_search = config["n_iter_random_search"]

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
        qrs_extraction_window=qrs_extraction_window,
        combine_features=True
    )

    # split data based on patients    
    assert X is not None, "X is None"
    assert y is not None, "y is None"
    assert groups is not None, "groups is None"

    # label encode y
    le = LabelEncoder()
    y_encoded = np.array(le.fit_transform(y))
    X_train, X_test, y_train_encoded, y_test_encoded, group_train, _ = ml_utils.patient_wise_split(X, y_encoded, groups)    

    # check for oversampling
    if oversample_bool:
        X_train, y_train_encoded = ml_utils.oversample(X_train = X_train,y_train = y_train_encoded)

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
        # save label encoding
        ml_utils.save_label_encoding(label_encoding=le)
        print("Starting training...")   
        # check if grid search is true for model     
        grid_search = True if search_CV_params else False
        clf = ml_utils.fit_pipeline(clf=clf,  
                                    X_train=X_train, 
                                    y_train=y_train_encoded, 
                                    classifier_name=classifier_name, 
                                    grid_search=grid_search, 
                                    grid_params=search_CV_params,
                                    cv_split_param=cv_split_param,
                                    n_iter_random_search=n_iter_random_search,
                                    groups_train=group_train)
        
        # check if grid search is valid
        if search_CV_params:
            pass
        print("Finished training.")
        # get prediction
        y_pred_encoded = clf.predict(X_test)

        # transform back to labels
        y_train = np.array(le.inverse_transform(y_train_encoded))
        y_test = np.array(le.inverse_transform(y_test_encoded))
        y_pred = np.array(le.inverse_transform(y_pred_encoded))

        # compute macro f1
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

# * - shared ml utilities for training and evaluation - *
import os
import tempfile
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib


from arrhythmia_ml import file_utils, preprocess, extract_features


# ── mlflow ────────────────────────────────────────────────────────────────────

def setup_mlflow(config: dict):
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment_name"])


def get_run_by_name(experiment_name: str, run_name: str):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No run named '{run_name}' found.")

    return runs[0]


def save_test_data(X_test: np.ndarray, y_test: np.ndarray, y_train: np.ndarray):
    """Save test arrays as an MLflow artifact so eval.py can load them without joblib."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_data.npz")
        np.savez(path, X_test=X_test, y_test=y_test, y_train=y_train)
        mlflow.log_artifact(path, artifact_path="test_data")

def save_label_encoding(label_encoding):
    with tempfile.TemporaryDirectory() as tmp:
        le_path = os.path.join(tmp, "label_encoder.joblib")
        joblib.dump(label_encoding, le_path)
        mlflow.log_artifact(le_path, artifact_path="label_encoder")

def load_test_data(run_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    client = mlflow.tracking.MlflowClient()
    artifacts_path = client.download_artifacts(run_id, "test_data")
    data = np.load(f"{artifacts_path}/test_data.npz", allow_pickle=True)
    return data["X_test"], data["y_test"], data["y_train"]

def load_label_encoder(run_id: str):
    client = mlflow.tracking.MlflowClient()
    artifacts_path = client.download_artifacts(run_id, "label_encoder")
    return joblib.load(os.path.join(artifacts_path, "label_encoder.joblib"))

def load_model(run_id: str):
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


# ── feature matrix ────────────────────────────────────────────────────────────

def build_feature_matrix(
    raw_data_path: str,
    participant_ids: list[str],
    bandpass_window: list,
    wave_extraction_window: list,
    local_rr_mean_beat_window: int,
    wavelet_decomp_level: int,
    compute_only: list[str],
    keep_labels: np.ndarray,
    combine_features:bool,
    qrs_extraction_window: list[int] | None = None,
) ->  tuple[np.ndarray | None, ...]:

    X_wfms_all, X_rr_all, X_qrs_all, X_wavelets_all, y_all, g_all = [], [], [], [], [], []

    for pid in participant_ids:
        signal, fs, _, r_peaks, labels = file_utils.load_raw_participant_data(
            raw_data_path=raw_data_path, participant_id=pid
        )

        chan = 0
        raw_sig = signal[:, chan]
        sig = preprocess.notch_filter_1d(
            preprocess.bandpass_1d(raw_sig, fs=fs, low=bandpass_window[0], high=bandpass_window[1]),
            fs=fs,
            freq=50,
        )

        X_wfm_pid, X_rr_pid, X_qrs_pid, X_wavelet_pid = extract_features.return_commbined_feature_matrix(
            ecg_signal=sig,
            r_peaks=r_peaks,
            fs=fs,
            window_start_ms=wave_extraction_window[0],
            window_end_ms=wave_extraction_window[1],
            local_rr_beat_window=local_rr_mean_beat_window,
            wavelet_decomp_level=wavelet_decomp_level,
            qrs_extraction_window=qrs_extraction_window,
            compute_only=compute_only,
        )

        y_pid = np.array(labels)
        g_pid = np.full(shape=(len(y_pid),), fill_value=pid, dtype=object)

        if X_wfm_pid is not None:
            X_wfms_all.append(X_wfm_pid)
        if X_rr_pid is not None:
            X_rr_all.append(X_rr_pid)
        if X_qrs_pid is not None:
            X_qrs_all.append(X_qrs_pid)
        if X_wavelet_pid is not None:
            X_wavelets_all.append(X_wavelet_pid)
        
        y_all.append(y_pid)
        g_all.append(g_pid)


    y = np.concatenate(y_all)
    groups = np.concatenate(g_all)
    # filter to desired labels
    mask = np.isin(y, keep_labels)

    # mask groups and labels
    y, groups = y[mask], groups[mask]

    # stack whichever feature blocks were computed if needed
    if combine_features:
        parts = []
        if X_wfms_all:
            parts.append(np.vstack(X_wfms_all))
        if X_rr_all:
            parts.append(np.vstack(X_rr_all))
        if X_qrs_all:
            parts.append(np.vstack(X_qrs_all))
        if X_wavelets_all:
            parts.append(np.vstack(X_wavelets_all))

    # append all features horizontally next to each other
        X = np.hstack(parts)
        X = X[mask]

        return X, y, groups
    else:
        X_wfm = np.vstack(X_wfms_all)[mask] if X_wfms_all else None
        X_rr = np.vstack(X_rr_all)[mask] if X_rr_all else None
        X_qrs = np.vstack(X_qrs_all)[mask] if X_qrs_all else None
        X_wavelet = np.vstack(X_wavelets_all)[mask] if X_wavelets_all else None
        return X_wfm, X_rr, X_qrs, X_wavelet, y, groups



# ── train / split / oversample ─────────────────────────────────────────────────────────────

def patient_wise_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, ...]:

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    group_train, group_test = groups[train_idx], groups[test_idx]

    return X_train, X_test, y_train, y_test, group_train, group_test

def oversample(X_train:np.ndarray, y_train:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """_summary_

    Args:
        X_train (np.ndarray): _description_
        y_train (np.ndarray): _description_

    Returns:
        tuple[np.ndarray,np.ndarray]: _description_
    """
    sm = SMOTE()
    X_train_oversamp, y_train_oversamp = sm.fit_resample(X=X_train,y=y_train)

    return X_train_oversamp, y_train_oversamp


def build_pipeline(classifier_name: str, classifier_params: dict) -> Pipeline:
    """
    Build a StandardScaler + classifier pipeline.
    Add new classifiers here as the project grows.

    Supported options (set under experiments.<exp>.classifier in config.yaml):
        'logreg'  -> LogisticRegression
        'gboost'  -> GradientBoostingClassifier
    """
    if classifier_name == "logreg":
        estimator = LogisticRegression(
            max_iter=classifier_params.get("max_iter", 2000),
            class_weight="balanced",
            solver=classifier_params.get("solver", "saga"),
            verbose=classifier_params.get("verbose",False)
        )

    elif classifier_name == "gboost":
        estimator = GradientBoostingClassifier(
            n_estimators=classifier_params.get("n_estimators", 100),
            n_iter_no_change=classifier_params.get("n_iter_no_change", 10),
            validation_fraction=classifier_params.get("validation_fraction", 0.1),
            verbose=1,
        )
    elif classifier_name == 'xgboost':
        estimator = XGBClassifier(
            n_estimators=classifier_params.get("n_estimators",100),
            colsample_bylevel=classifier_params.get("colsample_bylevel",0.7),
            tree_method=classifier_params.get("tree_method",'hist'),
            device=classifier_params.get("device",'cpu')
        )

    else:
        raise ValueError(
            f"Unknown classifier: '{classifier_name}'. Add it to build_pipeline() in ml_utils.py."
        )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])

    return clf


def fit_pipeline(
    clf: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_name: str,
    grid_search:bool=False,
    grid_params:dict={},
    cv_split_param:int=5,
    n_iter_random_search=30,
    groups_train:np.ndarray | None =None,
) -> Pipeline:
    # classifiers that cannot use class_weight natively need sample_weight at fit time
    needs_sample_weight = classifier_name in ["gboost","xgboost"]
    if needs_sample_weight:
        sample_weights = compute_sample_weight("balanced",y_train)
    else:
        sample_weights = None

    if grid_search:
        assert grid_params is not None, 'Grid params are empty. Check config'
        # define cross validation splits
        cv = GroupKFold(n_splits=cv_split_param)
        search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=grid_params,
            n_iter=n_iter_random_search,
            cv=cv,
            scoring='f1_macro',
            random_state=42,
            verbose=2,
            n_jobs=-1
        )
        # make sure params match before search
        assert X_train.shape[0] == len(y_train), f"Shape of X_train {len(X_train)} and y_train {len(y_train)} dont match"
        assert len(y_train) == len(groups_train), f"Shape of y_train {len(y_train)} and groups_train {len(groups_train)} dont match"

        search.fit(X_train, y_train, groups=groups_train, clf__sample_weight=sample_weights)
        clf  = search.best_estimator_
    else:
        clf.fit(X_train, y_train, clf__sample_weight=sample_weights)


    return clf


# ── metrics ───────────────────────────────────────────────────────────────────

def log_metrics(y_test: np.ndarray, y_pred: np.ndarray, keep_labels: np.ndarray):
    """Log macro F1 and per-class F1 to the active MLflow run."""
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    mlflow.log_metric("macro_f1", macro_f1)

    for label in keep_labels:
        mlflow.log_metric(f"f1_{label}", f1_score(y_test, y_pred, labels=[label], average="macro"))

    return macro_f1

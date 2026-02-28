from arrhythmia_ml import file_utils, preprocess, extract_features
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib


# load all config params
config = file_utils.load_config()
raw_data_path = config["raw_data_path"]
models_path, features_path = config["models_path"], config["features_path"]
bandpass_window = config["general_bandpass"]
wave_extraction_window = config["wfm_extraction_window"]
max_training_iter = config["max_iter"]
local_rr_mean_beat_window = config["local_rr_mean_beat_window"]
compute_only = ["waves", "interval_related"]

participant_ids = file_utils.get_participant_ids(raw_data_path=raw_data_path)

X_wfms_all, X_rr_all, y_all, g_all = [], [], [], []

for pid in participant_ids:
    signal, fs, channels, r_peaks, labels = file_utils.load_raw_participant_data(
        raw_data_path=raw_data_path, participant_id=pid
    )

    chan = 0
    raw_sig = signal[:, chan]
    sig = preprocess.notch_filter_1d(
        preprocess.bandpass_1d(
            raw_sig, fs=fs, low=bandpass_window[0], high=bandpass_window[1]
        ),
        fs=fs,
        freq=50,
    )

    X_wfm_pid, X_rr_pid = extract_features.return_commbined_feature_matrix(
        ecg_signal=sig,
        r_peaks=r_peaks,
        fs=fs,
        window_start_ms=wave_extraction_window[0],
        window_end_ms=wave_extraction_window[1],
        local_rr_beat_window=local_rr_mean_beat_window,
        compute_only = compute_only
    )

    y_pid = np.array(labels)
    g_pid = np.full(shape=(len(y_pid),), fill_value=pid, dtype=object)

    X_wfms_all.append(X_wfm_pid)
    X_rr_all.append(X_rr_pid)

    y_all.append(y_pid)
    g_all.append(g_pid)

X_wfm = np.vstack(X_wfms_all)
X_rr = np.vstack(X_rr_all)

X = np.hstack((X_wfm, X_rr))
y = np.concatenate(y_all)
groups = np.concatenate(g_all)

# keep only desired labels
keep_labels = np.array(["N", "A", "V"])
mask = np.isin(y, keep_labels)
X, y, groups = X[mask], y[mask], groups[mask]

# patient-wise split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("Starting training")
clf = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                max_iter=max_training_iter, class_weight="balanced", solver="saga"
            ),
        ),
    ]
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Finished training")
joblib.dump(clf, f"{models_path}/Xwaves_Xrr_logreg_patientwise_nav.joblib")
joblib.dump(
    {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    },
    f"{features_path}/Xwaves_Xrr_patientwise_nav_features.joblib",
)

print("models and features saved")

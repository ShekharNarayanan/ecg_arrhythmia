from arrhythmia_ml import file_utils, preprocess, extract_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# LOAD PARTICIPANT DATA
config = file_utils.load_config()
raw_data_path = config["raw_data_path"]
participant_ids = file_utils.get_participant_ids(raw_data_path=raw_data_path)
test_id = participant_ids[0]

X_all = []
y_all = []

for pid in participant_ids:  # start with 10
    signal, fs, channels, r_peaks, labels = file_utils.load_raw_participant_data(
        raw_data_path=raw_data_path, participant_id=pid
    )

    chan = 0
    raw_sig = signal[:, chan]
    sig = preprocess.notch_filter_1d(preprocess.bandpass_1d(raw_sig, fs=fs, low=0.5, high=30.0), fs=fs, freq=50)

    X_waveforms = extract_features.return_commbined_feature_matrix(
        ecg_signal=sig, peaks=r_peaks, fs=fs, window_start_ms=-200, window_end_ms=400
    )

    X_all.append(X_waveforms)
    y_all.append(np.array(labels))

X = np.vstack(X_all)
y = np.concatenate(y_all)

# Count occurrences of each label
keep_labels = np.array(["N", "A", "V"])

mask = np.isin(y, keep_labels)

X = X[mask]
y = y[mask]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# model
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", verbose=True, solver="saga"),
    )
])

clf.fit(X_train, y_train)

# eval
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))


import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from arrhythmia_ml import file_utils

# load config params
config = file_utils.load_config()
waves_model  = config["waves_model"]
waves_features = config["waves_model_features"]

# load model
clf = joblib.load(waves_model)

# load cached features
data = joblib.load(waves_features)
X_test = data["X_test"]
y_test = data["y_test"]

# predict
y_pred = clf.predict(X_test)

# print report
print(classification_report(y_test, y_pred, digits=3))

# plot confusion matrix (recall view)
labels = ["N", "A", "V"]

macro_f1 = f1_score(y_test, y_pred, average="macro")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=labels,
    display_labels=labels,
    normalize="true",
)

plt.title(f"Patient-wise Confusion Matrix | Macro F1 = {macro_f1:.3f}")
plt.show()
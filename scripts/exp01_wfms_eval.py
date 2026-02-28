import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from arrhythmia_ml import file_utils
import numpy as np

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
y_train = data["y_train"]

# predict
y_pred = clf.predict(X_test)

# get params relevant for eval plots
class_labels, counts = np.unique(y_train, return_counts=True)
perc = (counts / counts.sum()) * 100
macro_f1 = f1_score(y_test, y_pred, average="macro")

# define plot params 
fig, axes = plt.subplots(
    1,2,
    figsize=(10,4)
)

# Confusion matrix 
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=class_labels,
    display_labels=class_labels,
    normalize="true",
    ax=axes[0],
)

axes[0].set_title(f"Patient-wise Confusion Matrix | Macro F1 = {macro_f1:.3f}")

#  Class imbalance (training set) 
ax = axes[1]
bars = ax.bar(class_labels, perc, width=0.4, color="#E65C00")

ax.set_title("Class Distribution (Training Set)", fontsize=12 )
ax.set_xlabel("Class", )
ax.set_ylabel("Percentage (%)" )
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
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from arrhythmia_ml import file_utils
import seaborn as sns

# get config
config = file_utils.load_config()
mlflow_export_path = config["mlflow_exported"]

# load data
macro_f1_csv = pd.read_csv(f"{mlflow_export_path}/macro_f1.csv")

# macro f1 bar charts
model_names = macro_f1_csv["Run"]
macro_f1_data = macro_f1_csv["macro_f1"]

colors = sns.color_palette("Set2", len(model_names))
plt.figure(figsize=(10,3))
for i,(model,f1) in enumerate(zip(model_names,macro_f1_data)):
    plt.barh(y=model, width=f1, color=colors[i])
plt.xlabel("macro_f1")
smote_idx = list(model_names).index("exp03_waves_rr_gboost_smote")
smote_f1 = macro_f1_data.iloc[smote_idx]
plt.annotate("oversampled", xy=(smote_f1, smote_idx), xytext=(smote_f1+0.05, smote_idx+0.8), arrowprops=dict(arrowstyle="->"))
plt.tight_layout()
plt.show()
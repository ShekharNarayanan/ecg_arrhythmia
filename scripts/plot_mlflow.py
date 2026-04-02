import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from arrhythmia_ml import file_utils
import seaborn as sns
import yaml
from pathlib import Path
# get config
with open(Path(__file__).resolve().parents[1] / "config.yaml", "r") as file:
        config = yaml.safe_load(file)
mlflow_export_path = config["mlflow_exported"]

# load data
type = 'comparison'
macro_f1_csv = pd.read_csv(f"{mlflow_export_path}/macro_f1_{type}.csv")

# macro f1 bar charts
model_names = macro_f1_csv["Run"]
macro_f1_data = macro_f1_csv["macro_f1"]

colors = sns.color_palette("Set2", len(model_names))
plt.figure(figsize=(10,3))
for i,(model,f1) in enumerate(zip(model_names,macro_f1_data)):
    plt.barh(y=model, width=f1, color=colors[i])
plt.xlabel("macro_f1")
plt.tight_layout()
plt.show()
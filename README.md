# ECG Data Exploration, Preprocessing and Machine Learning
This project focuses on **exploring/analyzing ECG data** with literature-backed methods with the **[MIT-BIH Arrhythmia Database (PhysioNet)](https://physionet.org/content/mitdb/1.0.0/)**. There is also focus on making these scripts more production ready in terms of how other teams can use the code developed. 

**NOTE**: This project is now concluded. For a deep dive into methods and insights please visit [this](https://medium.com/@shekharnarayanan833/the-downside-to-clarity-a-machine-learning-only-approach-to-arrhythmia-detection-63074e96851a?postPublishedType=initial) article.

## Index
**Project Information**
1. [Preprocessing](#preprocessing)
2. [Data Analysis and Machine Learning](#data-analysis-and-machine-learning)
3. [MLOps](#ml-ops)
4. [Visuals](#visuals)

**Setup and Usage**
1. [Setup Instructions](#setup)
   - [Option A: Docker (Recommended)](#option-a-docker-recommended)
   - [Option B: Local Setup](#option-b-local-setup)
2. [Usage](#usage)
   - [Using Docker](#using-docker)
   - [Using Local Setup](#using-local-setup)
3. [Contact](#contact)

---

## Preprocessing
- [x] 0.5 - 30 Hz Filtering
- [x] Notch filtering
- [x] Detrend and baseline correction
- [x] R-peak Detection
    - [x] Write modules for Pam-Tompson QRS complex detection
    - [x] Calibrate peaks after detection
    - [x] Validate detected peaks with annotations as ground truth

### Literature used:
- [Rahul Kher (2019) Signal Processing Techniques for Removing Noise from ECG Signals. J Biomed Eng 1: 1-9](https://www.jscholaronline.org/articles/JBER/Signal-Processing.pdf)
- https://martager.github.io/bbsig/ecg-preprocessing/

## Feature Extraction and Engineering
- [x] **Extract R peak waveforms and other key features**
- [x] **Build table with labels from annotations**
- [x] **Extract RR interval features**: pre/post RR, delta RR, RR ratio, local RR mean, deviation from local mean
- [x] **Extract QRS morphology features**: Q-R interval, R-S interval, Q-R amplitude, R-S amplitude
- [x] **Extract wavelet features and use them with XGBoost**

### Literature used:
- https://inass.org/wp-content/uploads/2022/12/2023043016-2.pdf

## data Analysis and Machine Learning
- [x] **Logistic regression with R waveforms only**: Macro F1 = 0.41
- [x] **Logistic regression with waveforms + RR features**: Macro F1 = 0.44
- [x] **Gradient boosting with waveforms + RR features**: Macro F1 = 0.529
- [x] **SMOTE oversampling**: Macro F1 = 0.520 (dropped: slowed training, no improvement)
- [x] **Feature ablation**: waves + RR + QRS outperforms subsets; QRS features add value
- [x] **XGBoost with waveforms + RR + QRS features**: Macro F1 = 0.577
- [x] **XGBoost hyperparameter optimization**: randomized search with patient-wise CV
- [x] **RR irregularity feature (coefficient of variation of RR intervals)**

## ML Ops
- [x] **MLflow experiment tracking**: all runs tracked with metrics, parameters, and artifacts for reproducibility and comparison across experiments
- [x] **Config-driven experiments**: new experiments require only a config.yaml entry, no code changes
- [x] **Label encoding and artifact logging**: LabelEncoder saved as MLflow artifact for consistent eval
- [x] **Finalized config file for ablation use in app**

## Visuals
- [x] Create initial relevant visuals for N vs A vs V beats:
    - [x] Waveform shape
    - [x] pre_rr vs post_rr intervals scatter distribution
    - [x] rr_irregularity violin plot
    - [x] QRS beat variation violin
    - [x] Change of rr_irregularity over time
    - [x] Feature correlation matrix to show redundancy

---

## Setup

### Option A: Docker (Recommended)

No Python, no uv, no dependency setup required :just Docker.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your platform
2. Pull the image:
```bash
docker pull theopensourceguy/arrhythmia_ml
```
3. Download the raw data from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) and extract it into `./data/raw` in whatever folder you'll be running commands from

That's it — see [Using Docker](#using-docker) for how to run scripts.

---

### Option B: Local Setup

1. Install the following before going further:
   1. Git
   2. [uv](https://docs.astral.sh/uv/)
   3. Raw data from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) — extract to `data/raw`

2. Clone the repository:
```bash
git clone https://github.com/ShekharNarayanan/arrhythmia_ml.git
```

3. Navigate to the repository:
```bash
cd arrhythmia_ml
```

4. Install dependencies:
```bash
uv sync
```

---

## Usage

### Using Docker

> Always run Docker commands from the same directory. Your data should be in `./data/raw` and MLflow runs will be saved to `./mlruns` in that directory.

**Train a model:**
```bash
docker run -v ./data:/app/data -v ./mlruns:/app/mlruns theopensourceguy/arrhythmia_ml uv run python -m scripts.train --exp <exp_name>
```

**Evaluate results:**
```bash
docker run -v ./data:/app/data -v ./mlruns:/app/mlruns theopensourceguy/arrhythmia_ml uv run python -m scripts.eval --exp <exp_name>
```

**Track experiments in MLflow UI:**
```bash
docker run -v ./mlruns:/app/mlruns -p 5000:5000 theopensourceguy/arrhythmia_ml uv run mlflow ui --host 0.0.0.0
```
Then open `http://localhost:5000` in your browser.

**Validate peak detection:**
```bash
docker run -v ./data:/app/data -v ./peak_detection_plots:/app/peak_detection_plots theopensourceguy/arrhythmia_ml uv run python validate_detected_peaks.py
```

**Visualize features:**
```bash
docker run -v ./data:/app/data -v ./feature_plots:/app/feature_plots theopensourceguy/arrhythmia_ml uv run python visualize_features.py
```

---

### Using Local Setup

1. Make changes in `config.yaml` to specify your settings (beats to classify, preprocessing parameters, ML experiments etc.)

2. Train a model:
```bash
uv run python -m scripts.train --exp <insert exp from config>
```
This will create the `mlruns` folder. MLflow will track your experiments using it.

3. Evaluate results:
```bash
uv run python -m scripts.eval --exp <insert exp from config>
```

4. Track experiments in MLflow UI:
```bash
uv run mlflow ui
```

5. Validate peak detection:
```bash
uv run python validate_detected_peaks.py
```

6. Visualize features:
```bash
uv run python visualize_features.py
```

---

## Contact
For any questions, you can contact me on [LinkedIn](https://www.linkedin.com/in/shekharnarayanan/).

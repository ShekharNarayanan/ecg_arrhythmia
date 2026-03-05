# ECG Data Exploration, Preprocessing and Machine Learning
This project focuses on **exploring/analyzing ECG data** with literature-backed methods with the **[MIT-BIH Arrhythmia Database (PhysioNet)](https://physionet.org/content/mitdb/1.0.0/)**. There is also focus on making these scripts more production ready in terms of how other teams can use the code developed. 
P.S: Setup and usage instructions coming soon
## Current stage => [Feature engineering / Machine Learning](#data-analysis-and-machine-learning):
![ECG example](media/readme/gboost_rr_smote.png)
Current insight: Gradient boosting with waveform + RR interval features + SMOTE oversampling reaches Macro F1 = 0.520. SMOTE balanced training classes to 33.3% each, but slightly hurt overall performance (from 0.529). A beat recall dropped (0.67 vs 0.76), suggesting current features don't separate A beats cleanly enough in feature space for synthetic samples to help. Next step: add RR irregularity and QRS morphology features.
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
### Literature used:
- https://inass.org/wp-content/uploads/2022/12/2023043016-2.pdf
## Data Analysis and Machine Learning
- [x] **Logistic regression with R waveforms only**: Macro F1 = 0.41
- [x] **Logistic regression with waveforms + RR features**: Macro F1 = 0.44
- [x] **Gradient boosting with waveforms + RR features**: Macro F1 = 0.529
- [x] **Address class imbalance (SMOTE)**: Macro F1 = 0.520 (slight drop; A beat precision impacted)
- [ ] **RR irregularity feature (coefficient of variation of RR intervals)**
- [ ] **Morphology features (QRS width, amplitude, skewness)**
## MLOps
- [x] **MLflow experiment tracking**: all runs tracked with metrics, parameters, and artifacts for reproducibility and comparison across experiments

## Data app to view results
- [ ]
- [ ]
- [ ]
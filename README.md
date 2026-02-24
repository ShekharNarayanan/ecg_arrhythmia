# ECG Data Exploration, Preprocessing and Machine Learning

This project focuses on **exploring/analyzing ECG data** with literature-backed methods with the **[MIT-BIH Arrhythmia Database (PhysioNet)](https://physionet.org/content/mitdb/1.0.0/)**. There is also focus on making these scripts more production ready in terms of how other teams can use the code developed. 

P.S: Setup and usage instructions coming soon

## Current stage => [Feature engineering / Machine Learning](#data-analysis-and-machine-learning):
![ECG example](media/readme/confusion_matrix_Xwaves.png)

Current insight: Got a baseline for beat classification (logistic regression) using only 'N', 'A' and 'V' beat types. Just using the R waveforms returns subpar performance. Time to add more features 😁



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
- [ ] **Extract RR and morphology features**

### Literature used:
- https://inass.org/wp-content/uploads/2022/12/2023043016-2.pdf

## Data Analysis and Machine Learning
- [x] **Use logistic regression only with R waveforms for base performance**
- [ ] **Deal with massive class imbalance**
- [ ]

## Data app to view results
- [ ]
- [ ]
- [ ]





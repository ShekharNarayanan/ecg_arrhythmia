# ECG Data Exploration, Preprocessing and Machine Learning

This project focuses on **exploring/analyzing ECG data** with literature-backed methods with the **[MIT-BIH Arrhythmia Database (PhysioNet)](https://physionet.org/content/mitdb/1.0.0/)**. There is also focus on making these scripts more production ready in terms of how other teams can use the code developed. 


![ECG example](media/readme/filtered_and_peaks.png)

## Current stage => **Feature engineering**:
- [ ] **Extract R peaks and build table with labels from annotations**

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
- [ ] **Extract R peaks and build table with labels from annotations**
- [ ]
- [ ]

## Data Analysis and Machine Learning
- [ ]
- [ ]
- [ ]

## Data app to view results
- [ ]
- [ ]
- [ ]





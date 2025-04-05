# Main Analysis

Scripts for EEG data preprocessing and autorejection analysis.

## Environment Setup:
conda env create -f environment.yml conda activate mne-dependencies

markdown
Copy
Edit

## Scripts:
- `preprocessing.py`: preprocess EEG data.
- `autoreject_testing_current_script.py`: test autorejection methods.

## Pipeline

1. Load + Configure EEG
   └─ Rename channels
   └─ Drop extras
   └─ Apply montage

2. Filter + Detrend
   └─ Bandpass 1–40 Hz
   └─ Detrend with scipy

3. ICA Artifact Removal
   └─ Fit ICA
   └─ Detect components correlated with FP1/FP2/F7/F8
   └─ Remove artifacts

4. Parse Events
   └─ Convert annotations to sample-wise labels
   └─ Group events into trials

5. Load Trial Metadata
   └─ Check if obstacles were present

6. Extract Epoch Windows
   └─ Epoch = [-2.5s to +1.5s around crossing]
   └─ Different logic for "Absent" vs "Present" trials

7. Time Warping
   └─ Extract crossing segment
   └─ Resample to median duration (simple DTW)
   └─ Align all trials

8. Save or Analyze
   └─ Plot, average, or feed to classifier

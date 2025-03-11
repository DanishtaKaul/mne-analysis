# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:41:49 2024

@author: Danishta Kaul

PRE-PROCESSING
1. Import and load the data from their respective files
2. Configure metadata
    a. Channel names and types
    b. Sampling rate
3. Band-pass filtering
4. Segregate data into 1s epochs
5. Run autoreject
6. ICA Processing
    a. Remove electro-ocular artifacts in the data
    b. Remove muscular artifacts
7. Interpolate bad channels with neighboring channels
8. Use auto rejection package to remove unusable data
9. Plot the data for a quality check
10. Save cleaned data into a file format for futher processing.
"""

import numpy as np
from scipy import signal
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet as time_frequency_morlet
import matplotlib.pyplot as plot
from preprocessing import *


def main():
    file_path = r"E:\PID 2 Light study\FIF EEG Data\PID 2 Ambient Expected.fif"
    montage_path = r"E:\Montage\Standard-10-10-Cap33_V6.loc"
    meta_info_path = r"E:\PID 2 Light study\PID 2 Unity Data\PID 2 Ambient Expected\S001\trial_results.csv"

    raw = load_and_configure_data(file_path, montage_path)

    filter_and_detrend_data(raw)
# =============================================================================
#     apply_ICA(raw)
# =============================================================================
    trial_events = create_trial_events(raw)
    create_epochs(raw, trial_events, meta_info_path)

    raw.plot()


main()

# Make epochs from events in raw to identify crossing times of obstacles


# Step 2: Initialize AutoReject
autoreject = AutoReject(
    verbose=True,
    random_state=42,
    n_jobs=-1,
    # Use Bayesian optimization for thresholds
    thresh_method='bayesian_optimization',
    cv=10  # Cross-validation for threshold estimation


)

# Step 3: Fit AutoReject on epochs
autoreject.fit(epochs)

# Step 4: Transform epochs using AutoReject
epochs_clean_ar, reject_log = autoreject.transform(epochs, return_log=True)

epochs_clean_ar.apply_baseline(
    baseline=(epochs_clean_ar.tmin, epochs_clean_ar.tmin + 1.0))

epochs_clean_ar.set_eeg_reference('average', projection=False)


# Step 5: Plot rejection log
reject_log.plot('horizontal')

# Step 6: Plot cleaned epochs
epochs_clean_ar.plot(
    n_epochs=len(epochs_clean_ar),  # Display all epochs at a time
    n_channels=32,  # Display 10 channels at a time
    scalings='auto',
    title='Further Cleaned Epochs'
)


# Define frequency range (3–40 Hz) and corresponding number of cycles
freqs = np.linspace(3, 40, 50)  # 50 frequencies between 3 and 40 Hz
# Number of cycles (adjust for desired time-frequency resolution)
n_cycles = freqs / 3

# Select the Fz channel
picks = mne.pick_channels(epochs_clean_ar.info['ch_names'], include=['Fz'])

# Compute time-frequency representation using Morlet wavelets
power = mne.time_frequency.tfr_morlet(
    epochs_clean_ar,
    freqs=freqs,
    n_cycles=n_cycles,
    picks=picks,
    return_itc=False,  # Return only power, not inter-trial coherence
)

# Plot the TFR for the Fz electrode
power.plot(
    picks='Fz',
    baseline=(-2.5, -1.5),  # Baseline period: 2.5 seconds before event onset
    # Baseline correction mode (subtract mean baseline power)
    mode='percent',
    title='Time-Frequency Plot (3–40 Hz, Fz Electrode, % Change from Baseline)',

    cmap='viridis'
)


# Fit ICA
# ica = mne.preprocessing.ICA(n_components=0.98, random_state=42)
# ica.fit(epochs_clean[good_epochs_mask], decim=3)

# print("ICA fitting complete!")

# ica.plot_components()


# # Initialize exclusion parameters
# ica.exclude = []  # Start with no excluded components
# num_excl = 0      # Number of excluded components
# max_ic = 2        # Maximum EOG-related components to exclude
# z_thresh = 3.5    # Initial z-score threshold for detecting bad components
# z_step = 0.05     # Step to reduce z-score threshold iteratively

# # Iteratively detect EOG-related components
# while num_excl < max_ic:
#     eog_indices, eog_scores = ica.find_bads_eog(
#         epochs_clean,  # Use the cleaned epochs
#         ch_name=['FP1', 'FP2', 'F7', 'F8'],  # Specify frontal channels for EOG
#         threshold=z_thresh  # Threshold for detection
#     )
#     num_excl = len(eog_indices)  # Count identified components
#     z_thresh -= z_step  # Decrease threshold for further iterations

# # Assign bad EOG components for exclusion
# ica.exclude = eog_indices

# # Print final threshold used for detection
# print('Final z threshold = ' + str(round(z_thresh, 2)))

# # Apply ICA to exclude identified components
# ica.apply(epochs_clean)

# # Visualize cleaned epochs (optional)
# epochs_clean.plot()

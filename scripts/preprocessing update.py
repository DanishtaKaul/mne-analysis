# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:12:29 2025

@author: danis
"""

import mne
import numpy as np
from scipy import signal

from scripts import logger
from mne_icalabel import label_components
from py_adjust import ADJUST

"""
 Apply bandpass (0-40Hz) and detrend data

 @Params raw
     The raw data file
# Adjust-Additional spatial/temporal artifact detection, eye movement
# Iclabel-Eye, Muscle, Heart, Line, Channel artifacta
"""


def filter_and_detrend_data(raw):
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
    raw._data = signal.detrend(raw.get_data(), axis=1)
    # average reference data
    raw.set_eeg_reference('average', projection=False)
    return raw


"""
 Apply bandpass and detrend data

 @Params raw
     The raw data file
"""
# z-score -How far is a value from the average, measured in standard deviations?
# mean correlation is the average correlation between each ICA component and the eye channels (FP1, FP2, F7, F8)
# Steps happening during ica-For each ICA component, compute the Pearson correlation with each eye channel (FP1, FP2, F7, F8).
# For each component, keep the highest absolute correlation value across these channels.
# Calculate the mean and standard deviation of these max correlation values across all components.
# For each component, compute its z-score: (max_corr - mean) / std.
# If a component's z-score is above the threshold (e.g., 3.5), flag it as an artifact (likely eye movement).


def apply_ICA(raw):

    ica = mne.preprocessing.ICA(
        n_components=29, random_state=42, method='fastica')  # random_state- fixed number 42 will alway give same result each time code is run on same data
    # decim=3 means that ICA uses every 3rd time point to learn what artifacts look like then removes those artifacts from full original EEG data
    ica.fit(raw, decim=3)

    # Start with no components excluded
    ica.exclude = []

    # Parameters for iterative detection, z thresh-ica components are compared to signals from fp1, fp2, f7, f8 and if the z score of correlation with these channels is more than 3.5 they are considered artifact
    max_ic = 8       # Maximum number of EOG components expected
    # Only components with a z-score higher than 3.5 (3.5 standard deviations above the mean correlation with eye channels) will be flagged as artifact.
    z_thresh = 3.5
    z_step = 0.05    # Each time through the loop, if not enough artifact components are found, lower the threshold by 0.05

    # While the max number of EOG components hasn't been reached, keep lowering threshold, find_bads_eog() compares each ICA component to the eye channels
    while len(ica.exclude) < max_ic:
        eog_inds, eog_scores = ica.find_bads_eog(
            raw,
            # Using frontal channels as EOG proxies
            ch_name=['FP1', 'FP2', 'F7', 'F8'],
            threshold=z_thresh
        )

        # Combine new findings with any already excluded
        # ica.exclude list contains the indices of all components that looked like eye artifacts
        ica.exclude = list(set(ica.exclude + eog_inds))

        print(f"Threshold: {z_thresh:.2f}, EOG components found: {eog_inds}")

        # Lower threshold for next iteration if still haven't reached max_ic
        z_thresh -= z_step

        # If no new components were found in this iteration, break early
        if len(eog_inds) == 0:
            break

    print(f"Final excluded EOG components: {ica.exclude}")

    # Remove (exclude) the identified EOG components from raw data
    ica.apply(raw)

    # Visual checks
    ica.plot_scores(eog_scores)
    ica.plot_components()

    return raw

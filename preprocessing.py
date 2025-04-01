# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:36:50 2025

@author: danis
"""

import numpy as np
from scipy import signal
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet as time_frequency_morlet
import matplotlib.pyplot as plot
import pandas as pd
import re

from preprocessing import *


"""
 Reads raw file, renames electrode channels, sets the montage

 @Params file_path
     The filepath to the raw file

 @Params montage_path
     The filepath to the montage file
"""


def load_and_configure_data(file_path, montage_path):
    raw = mne.io.read_raw_fif(file_path, preload=True)
    raw.set_channel_types({ch_name: 'eeg' for ch_name in raw.info['ch_names']})
    raw.rename_channels({
        'EE214-000000-000116-02-DESKTOP-E45KF45_0': 'FP1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_1': 'FPz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_2': 'FP2',
        'EE214-000000-000116-02-DESKTOP-E45KF45_3': 'F7',
        'EE214-000000-000116-02-DESKTOP-E45KF45_4': 'F3',
        'EE214-000000-000116-02-DESKTOP-E45KF45_5': 'Fz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_6': 'F4',
        'EE214-000000-000116-02-DESKTOP-E45KF45_7': 'F8',
        'EE214-000000-000116-02-DESKTOP-E45KF45_8': 'FC5',
        'EE214-000000-000116-02-DESKTOP-E45KF45_9': 'FC1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_10': 'FC2',
        'EE214-000000-000116-02-DESKTOP-E45KF45_11': 'FC6',
        'EE214-000000-000116-02-DESKTOP-E45KF45_12': 'M1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_13': 'T7',
        'EE214-000000-000116-02-DESKTOP-E45KF45_14': 'C3',
        'EE214-000000-000116-02-DESKTOP-E45KF45_15': 'Cz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_16': 'C4',
        'EE214-000000-000116-02-DESKTOP-E45KF45_17': 'T8',
        'EE214-000000-000116-02-DESKTOP-E45KF45_18': 'M2',
        'EE214-000000-000116-02-DESKTOP-E45KF45_19': 'CP5',
        'EE214-000000-000116-02-DESKTOP-E45KF45_20': 'CP1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_21': 'CP2',
        'EE214-000000-000116-02-DESKTOP-E45KF45_22': 'CP6',
        'EE214-000000-000116-02-DESKTOP-E45KF45_23': 'P7',
        'EE214-000000-000116-02-DESKTOP-E45KF45_24': 'P3',
        'EE214-000000-000116-02-DESKTOP-E45KF45_25': 'Pz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_26': 'P4',
        'EE214-000000-000116-02-DESKTOP-E45KF45_27': 'P8',
        'EE214-000000-000116-02-DESKTOP-E45KF45_28': 'POz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_29': 'O1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_30': 'Oz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_31': 'O2',
        'EE214-000000-000116-02-DESKTOP-E45KF45_32': 'N/A1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_33': 'N/A2',
    }
    )
    raw.drop_channels({'M1', 'M2', 'N/A1', 'N/A2'})
    montage = mne.channels.read_custom_montage(montage_path)
    raw.set_montage(montage)
    return raw


"""
 Apply bandpass and detrend data

 @Params raw
     The raw data file
"""


def filter_and_detrend_data(raw):
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
    signal.detrend(raw.get_data(), axis=1)


"""
 Apply bandpass and detrend data

 @Params raw
     The raw data file
"""


def apply_ICA(raw):

    ica = mne.preprocessing.ICA(
        n_components=29, random_state=42, method='fastica')
    ica.fit(raw, decim=3)

    # Start with no components excluded
    ica.exclude = []

    # Parameters for iterative detection, z thresh-ica components are compared to signals from fp1, fp2, f7, f8 and if the z score of correlation with these channels is more than 3.5 they are considered artifact
    max_ic = 8       # Maximum number of EOG components expected
    z_thresh = 3.5   # Initial high z-score threshold
    z_step = 0.05    # Step to decrease the threshold iteratively

    # While the max number of EOG components hasn't been reached, keep lowering threshold
    while len(ica.exclude) < max_ic:
        eog_inds, eog_scores = ica.find_bads_eog(
            raw,
            # Using frontal channels as EOG proxies
            ch_name=['FP1', 'FP2', 'F7', 'F8'],
            threshold=z_thresh
        )

        # Combine new findings with any already excluded
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


"""
    Create epochs 2.5s before crossing an obstacle and 1.5s after crossing

 @Params raw
     The raw data file
"""


def create_epochs(raw):
    return
    events, events_dict = mne.events_from_annotations(raw)

    # 1. Find the start and end events using substrings (start: "Right", "Toe", "Start", "Bounds" ! End: "Left", "Heel", "End", "Bounds")
    # 2. Put the start and end event codes into seperate arrays. Check both of these are the same size. If they are not, we have a misleading event and therefore the epoch runs over a trial (A code is invalid and should default to a value ~0.5seconds)
    # 3. Make an event array the same length as the start and end arrays, with each event having the start of the start (Right Heel Start BOunds) and the end of the event having the start of the end (Left Heel End Bounds)
    # 4. Finally, make the epochs with these events with -2.5s tmin and +1.5 tmax.

    start = []
    end = []

    for key, value in events_dict.items():
        if "RightFootToe" in key and "start" in key and "_ObstacleCrossingBounds" in key:
            start.append(value)
        if "LeftFootHeel" in key and "end" in key and "_ObstacleCrossingBounds" in key:
            end.append(value)
        # How to get the event
         # events[events[:, 2] == target_event_code]

    # Convert start and end sample indices to time (in seconds)
    start_times = np.array([events[events[:, 2] == event][0]
                           [0] / raw.info['sfreq'] for event in start])
    end_times = np.array([events[events[:, 2] == event][0]
                         [0] / raw.info['sfreq'] for event in end])

    smallest_array = start_times if len(
        start_times) < len(end_times) else end_times

    arr = np.concatenate((start_times, end_times))
    arr = np.sort(arr)

    new_events = []
    sfreq = raw.info['sfreq']
    for i in range(0, len(arr), 2):
        event_start = int(round(arr[i] * sfreq))
        try:
            event_end = int(round(arr[i + 1] * sfreq))
        except IndexError:

            event_end = int(round((arr[i] + 0.7) * sfreq))

        new_events.append([event_start, 0, 1])

    epochs = mne.Epochs(
        raw,
        events=new_events,
        event_id=1,
        tmin=-2.5,
        tmax=1.5,
        baseline=(None, 0),
        preload=True
    )

    epochs.plot(n_epochs=10, n_channels=10, scalings='auto', title='Epochs')


"""
# I need to group these events within an array with each element of this array tied to a trial.
# Within each trial, it should be the sequential occurance of said events aka: Trial start, Walk, Right foot Croosbounds etc...
"""

''


def create_trial_events(raw):
    """
    Create trial events using the absolute sample indices from the events array.
    Each event is stored as a dictionary with 'sample' (absolute sample index)
    and 'label' (annotation label).
    """
    # Get events array and mapping from annotations to event codes.
    events, event_id = mne.events_from_annotations(raw)
    
    # Invert event_id to map event code to label.
    code_to_label = {code: label for label, code in event_id.items()}
    
    trial_events = []
    current_trial = []
    
    # The events array is already sorted by sample index.
    for event in events:
        sample = event[0]              # Absolute sample index in raw data
        code = event[2]
        label = code_to_label.get(code, "")
        ev_dict = {"sample": sample, "label": label}
        
        # Use "Start Trial" to mark the beginning of a new trial.
        if "Start Trial" in label:
            if current_trial:
                trial_events.append(current_trial)
            current_trial = [ev_dict]
        else:
            current_trial.append(ev_dict)
    
    if current_trial:
        trial_events.append(current_trial)
    
    return trial_events


def create_epochs(raw, trialEvents, meta_info_path):
    """
    For each trial, extract crossing times using the absolute sample indices.
    The epoch is defined from (crossing start - 2.5 s) to (crossing end + 1.5 s)
    in absolute seconds on the raw file.
    
    Parameters:
      - raw: mne.io.Raw object.
      - trialEvents: List of trials, where each trial is a list of event dictionaries.
      - meta_info_path: Path to the metadata CSV file.
      
    Returns:
      - epochs: List of tuples, each containing (epoch_start, epoch_end) in seconds.
    """
    epochs = []
    sfreq = raw.info['sfreq']
    
    # Load metadata.
    meta = pd.read_csv(meta_info_path)
    
    def trial_is_absent(index):
        # Adjusting for 1-based indexing in metadata if needed.
        return meta.loc[index + 1, 'ExistanceLevel'] == 'Absent'
    
    for index, trial in enumerate(trialEvents):
        cs_time = None  # Crossing start time in seconds
        ce_time = None  # Crossing end time in seconds
        
        for event in trial:
            label = event['label']
            # Convert the absolute sample index to seconds.
            time_in_sec = event['sample'] / sfreq
            
            if trial_is_absent(index):
                # For absent trials, look for the midpoint crossing events.
                if "ObstacleCrossingMid start" in label:
                    cs_time = time_in_sec
                elif "ObstacleCrossingMid end" in label:
                    ce_time = time_in_sec
            else:
                # For trials with an obstacle, use the crossing bounds events.
                if "ObstacleCrossingBounds start" in label:
                    cs_time = time_in_sec
                elif "ObstacleCrossingBounds end" in label:
                    ce_time = time_in_sec
        
        # Only create an epoch if both crossing times were found.
        if cs_time is not None and ce_time is not None:
            # These boundaries are absolute times on the raw file.
            epoch_start = cs_time - 2.5
            epoch_end = ce_time + 1.5
            epochs.append((epoch_start, epoch_end))
    
    return epochs

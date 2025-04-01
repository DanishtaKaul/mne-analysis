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
import os
import numpy as np
from scipy import signal
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet as time_frequency_morlet
import matplotlib.pyplot as plot
from preprocessing import *

debug = False
experiments = [r"E:\PID 3", r"E:\PID 4", r"E:\PID 5"]
# experiment_root = r"E:\PID 3"
montage_path = r"E:\Montage\Standard-10-10-Cap33_V6.loc"


def load_multiple_experiments():
    for experiment in experiments:
        navigate_experiment(experiment)

# Navigate through an experiments file structure in the continuous matter of which the experiment ran


def navigate_experiment(experiment_root):
    eeg_dir = None
    unity_dir = None

    # Print the contents of experiment_root for debugging purposes.
    print(os.listdir(experiment_root))

    # Find the EEG and Unity directories.
    for item in os.listdir(experiment_root):
        item_path = os.path.join(experiment_root, item)
        if os.path.isdir(item_path):
            lower_item = item.lower()
            if 'fif' in lower_item:
                eeg_dir = item_path
            elif 'unity' in lower_item:
                unity_dir = item_path

    if eeg_dir is None:
        print("No EEG folder (with 'fif' in its name) found in experiment root.")
        return
    if unity_dir is None:
        print("No Unity folder (with 'unity' in its name) found in experiment root.")
        return

    # Define conditions.
    light_conditions = ['light', 'ambient', 'dark']
    obstacle_conditions = ['expected', 'unexpected']

    # Process each combination of conditions.
    for light in light_conditions:
        for obstacle in obstacle_conditions:

            # Search for the appropriate EEG file.
            eeg_file_path = None
            for file in os.listdir(eeg_dir):
                file_lower = file.lower()
                if file_lower.endswith('.fif') and (light in file_lower):
                    # For "expected", ensure "unexpected" is not present.
                    if obstacle == 'expected' and 'unexpected' in file_lower:
                        continue
                    # For "unexpected", ensure "unexpected" is present.
                    elif obstacle == 'unexpected' and 'unexpected' not in file_lower:
                        continue
                    eeg_file_path = os.path.join(eeg_dir, file)
                    break

            if eeg_file_path is None:
                print(f"No EEG file found for condition: {light} & {obstacle}")
                continue

            # Search for the appropriate Unity block folder.
            unity_block_dir = None
            for folder in os.listdir(unity_dir):
                folder_lower = folder.lower()
                folder_path = os.path.join(unity_dir, folder)
                if os.path.isdir(folder_path) and (light in folder_lower):
                    if obstacle == 'expected' and 'unexpected' in folder_lower:
                        continue
                    elif obstacle == 'unexpected' and 'unexpected' not in folder_lower:
                        continue
                    unity_block_dir = folder_path
                    break

            if unity_block_dir is None:
                print(
                    f"No Unity block folder found for condition: {light} & {obstacle}")
                continue

            # Locate the subfolder starting with 'S'.
            s_folder = None
            for folder in os.listdir(unity_block_dir):
                folder_path = os.path.join(unity_block_dir, folder)
                if os.path.isdir(folder_path) and folder.lower().startswith('s'):
                    s_folder = folder_path
                    break

            if s_folder is None:
                print(
                    f"No subfolder starting with 'S' found in Unity block folder for condition: {light} & {obstacle}")
                continue

            # Check if trial_results.csv exists.
            trial_results_path = os.path.join(s_folder, "trial_results.csv")
            if not os.path.exists(trial_results_path):
                print(
                    f"trial_results.csv not found in {s_folder} for condition: {light} & {obstacle}")
                continue

            # Process the block.

            if (debug):
                print("\n ====== \n")
                print(f"CURRENT EXPERIMENT: {experiment_root}")
                print(
                    f"Processing condition: {light.title()}, {obstacle.title()}")
                print(f"file_path: {eeg_file_path}")
                print(f"meta_info_path: {trial_results_path}")
                print("\n ====== \n")
            else:
                block_preprocessing(eeg_file_path, trial_results_path)


def block_preprocessing(file_path, meta_info_path):

    # Constant

    raw = load_and_configure_data(file_path, montage_path)

    # filter_and_detrend_data(raw)

    # apply_ICA(raw)
    trial_events = create_trial_events(raw)
    epochs = create_epochs(raw, trial_events, meta_info_path)
    print("\n ====== EPOCHS ====== \n")
    print(epochs)
    print("\n ====== \n")
    # raw.plot()


# load_multiple_experiments()
block_preprocessing(r"E:\PID 3\FIF\PID 3 LIGHT EXPECTED.fif",
                    r"E:\PID 3\UNITY\pid 3 light expected\S005\trial_results.csv")

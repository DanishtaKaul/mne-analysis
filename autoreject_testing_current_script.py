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

experiment_root = r"E:\backup\PID 2 Light study"

# Navigate through an experiments file structure in the continuous matter of which the experiment ran


def navigate_experiment(experiment_root):
    eeg_dir = None
    unity_dir = None

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

    light_conditions = ['light', 'ambient', 'dark']
    obstacle_conditions = ['expected', 'unexpected']

    for light in light_conditions:
        for obstacle in obstacle_conditions:
            eeg_file_path = None
            for file in os.listdir(eeg_dir):
                if file.lower().endswith('.fif') and (light in file.lower()) and (obstacle in file.lower()):
                    eeg_file_path = os.path.join(eeg_dir, file)
                    break

            if eeg_file_path is None:
                print(f"No EEG file found for condition: {light} & {obstacle}")
                continue

            unity_block_dir = None
            for folder in os.listdir(unity_dir):
                folder_path = os.path.join(unity_dir, folder)
                if os.path.isdir(folder_path) and (light in folder.lower()) and (obstacle in folder.lower()):
                    unity_block_dir = folder_path
                    break

            if unity_block_dir is None:
                print(
                    f"No Unity block folder found for condition: {light} & {obstacle}")
                continue

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

            trial_results_path = os.path.join(s_folder, "trial_results.csv")
            if not os.path.exists(trial_results_path):
                print(
                    f"trial_results.csv not found in {s_folder} for condition: {light} & {obstacle}")
                continue

            print(f"Processing condition: {light.title()}, {obstacle.title()}")
            block_preprocessing(eeg_file_path, trial_results_path)

# This function is called over each block within an experiment


def block_preprocessing(file_path, meta_info_path):

    print('\n ===== \n')
    print('block_preprocessing called')
    print(f"file_path: {file_path}")
    print(f"meta_info_path: {meta_info_path}")
    print('\n ===== \n')
    return

    montage_path = r"E:\Montage\Standard-10-10-Cap33_V6.loc"  # Constant

    file_path = r"E:\PID 2 Light study\FIF EEG Data\PID 2 Ambient Expected.fif"  # EEG data
    meta_info_path = r"E:\PID 2 Light study\PID 2 Unity Data\PID 2 Ambient Expected\S001\trial_results.csv"

    raw = load_and_configure_data(file_path, montage_path)

    filter_and_detrend_data(raw)

    apply_ICA(raw)
    trial_events = create_trial_events(raw)
    create_epochs(raw, trial_events, meta_info_path)

    raw.plot()

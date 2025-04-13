import numpy as np
import pandas as pd
import mne
from scipy import signal
from tslearn.metrics import dtw_path
from config import debug, pre_crossing_sec
from scripts import logger
import matplotlib.pyplot as plt  # Needed for debug plotting

def align_epochs_with_dtw(raw, trialEvents, meta_info_path):
    """
    Create time-aligned epochs using Dynamic Time Warping (DTW).

    Parameters:
        raw (mne.io.Raw): Continuous EEG recording
        trialEvents (list): List of trials, each containing event dicts
        meta_info_path (str): Path to Unity metadata CSV

    Returns:
        aligned_epochs (mne.EpochsArray): DTW-aligned epochs
    """
    sfreq = raw.info['sfreq']
    meta = pd.read_csv(meta_info_path)

    crossing_epochs = []
    event_list = []

    def trial_is_absent(index):
        return meta.loc[index + 1, 'ExistanceLevel'] == 'Absent'

    # ----------------------------
    # Step 1: Extract Epochs
    # Search each trial for the crossing start/end times (cs_time, ce_time)
    # Use pre_crossing_sec and +1.5 s to define the window
    # Pull out EEG data from the raw file for that segment
    # Calculate relative offsets to mark where the crossing happens inside the window
    # ----------------------------
    for index, trial in enumerate(trialEvents):
        cs_time, ce_time = None, None  

        for event in trial:
            label = event['label']
            time_in_sec = event['sample'] / sfreq

            if trial_is_absent(index):
                if "ObstacleCrossingMid start" in label:
                    cs_time = time_in_sec
                elif "ObstacleCrossingMid end" in label:
                    ce_time = time_in_sec
            else:
                if "ObstacleCrossingBounds start" in label:
                    cs_time = time_in_sec
                elif "ObstacleCrossingBounds end" in label:
                    ce_time = time_in_sec

        if cs_time is not None and ce_time is not None:
            start_sample = int((cs_time - pre_crossing_sec) * sfreq)
            end_sample = int((ce_time + 1.5) * sfreq)

            data, _ = raw[:, start_sample:end_sample]
            cs_offset = int(pre_crossing_sec * sfreq)
            ce_offset = int((ce_time - cs_time + pre_crossing_sec) * sfreq)

            crossing_epochs.append({
                'data': data,
                'crossing_start': cs_offset,
                'crossing_end': ce_offset
            })

            event_list.append([int(cs_time * sfreq), 0, 1])

    if not crossing_epochs:
        raise ValueError("No valid crossing epochs found")

    # ----------------------------
    # Step 2: Choose a Reference Crossing
    # ----------------------------
    median_idx = np.argsort([
        ep['crossing_end'] - ep['crossing_start']
        for ep in crossing_epochs
    ])[len(crossing_epochs) // 2]

    reference = crossing_epochs[median_idx]
    reference_crossing = reference['data'][:, reference['crossing_start']:reference['crossing_end']]
    reference_duration = reference_crossing.shape[1]

    # ----------------------------
    # Step 3: Align Each Epoch to Reference
    # ----------------------------
    aligned_data = []

    for epoch in crossing_epochs:
        pre_data = epoch['data'][:, :epoch['crossing_start']]
        crossing_data = epoch['data'][:, epoch['crossing_start']:epoch['crossing_end']]
        post_data = epoch['data'][:, epoch['crossing_end']:]

        aligned_crossing = np.zeros((crossing_data.shape[0], reference_duration))

        for ch_idx in range(crossing_data.shape[0]):
            ref = reference_crossing[ch_idx][:, np.newaxis]
            target = crossing_data[ch_idx][:, np.newaxis]

            path, _ = dtw_path(ref, target)

            warped = np.array([target[j][0] for i, j in path])
            warped_resampled = signal.resample(warped, reference_duration)

            aligned_crossing[ch_idx] = warped_resampled

        aligned_epoch = np.hstack((pre_data, aligned_crossing, post_data))
        aligned_data.append(aligned_epoch)

    # ----------------------------
    # Step 4: Create MNE Epochs
    # ----------------------------
    events = np.array(event_list)
    info = raw.info

    if debug:
        # Plot one example comparison
        plt.plot(crossing_data[0], label='Original')
        plt.plot(aligned_crossing[0], label='DTW Aligned')
        plt.legend()
        plt.title("Channel 0: Original vs DTW-Aligned Crossing")
        plt.show()

    aligned_epochs = mne.EpochsArray(
        np.stack(aligned_data),  # Ensures uniform shape
        info=info,
        tmin=-pre_crossing_sec,
        events=events,
        event_id={1: 1}
    )

    return aligned_epochs

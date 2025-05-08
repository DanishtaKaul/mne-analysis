import numpy as np
import mne
from scipy import signal
from tslearn.metrics import dtw_path
from config import debug, pre_crossing_sec, post_crossing_sec
from collections import Counter
import matplotlib.pyplot as plt


def pad_or_crop_epoch(epoch_data, target_length):
    """
    Pad or crop an EEG epoch to match a fixed length.

    Parameters:
        epoch_data (ndarray): EEG data of shape (n_channels, n_samples)
        target_length (int): Desired number of time samples

    Returns:
        ndarray: Epoch with shape (n_channels, target_length)
    """
    n_channels, current_length = epoch_data.shape

    if current_length == target_length:
        return epoch_data
    elif current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(epoch_data, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return epoch_data[:, :target_length]


def time_warping(raw, crossing_epochs, kept_indicies):
    # Keep only the epochs that passed AutoReject
    epochs = [crossing_epochs[i] for i in kept_indicies]

    """
    Apply Dynamic Time Warping (DTW) to align crossing segments across EEG epochs.

    Parameters:
        raw (mne.io.Raw): The original continuous EEG data (for metadata).
        epochs (list of dict): Each dict contains:
            - 'data': 2D array of EEG (n_channels x n_samples)
            - 'crossing_start': sample index in data where crossing begins
            - 'crossing_end': sample index in data where crossing ends

    Returns:
        mne.EpochsArray: DTW-aligned and shape-normalized EEG epochs
    """

    # Step 1: Choose a median-duration crossing as the DTW reference
    crossing_lengths = [ep['crossing_end'] - ep['crossing_start'] for ep in epochs]
    median_idx = np.argsort(crossing_lengths)[len(crossing_lengths) // 2]
    reference = epochs[median_idx]
    reference_crossing = reference['data'][:, reference['crossing_start']:reference['crossing_end']]
    reference_duration = reference_crossing.shape[1]

    aligned_data = []  # List to hold final aligned epochs

    for epoch in epochs:
        # Segment into pre-crossing, crossing, and post-crossing parts
        pre_data = epoch['data'][:, :epoch['crossing_start']]
        crossing_data = epoch['data'][:, epoch['crossing_start']:epoch['crossing_end']]
        post_data = epoch['data'][:, epoch['crossing_end']:]

        # Prepare container for aligned crossing segment
        aligned_crossing = np.zeros((crossing_data.shape[0], reference_duration))

        # Align each EEG channel independently using DTW
        for ch_idx in range(crossing_data.shape[0]):
            ref = reference_crossing[ch_idx][:, np.newaxis]     # shape: (n_samples, 1)
            target = crossing_data[ch_idx][:, np.newaxis]       # shape: (n_samples, 1)
            path, _ = dtw_path(ref, target)
            warped = np.array([target[j][0] for i, j in path])
            warped_resampled = signal.resample(warped, reference_duration)
            aligned_crossing[ch_idx] = warped_resampled

        # Combine aligned segments with unwarped pre/post segments
        aligned_epoch = np.hstack((pre_data, aligned_crossing, post_data))
        aligned_data.append(aligned_epoch)

    # Step 2: Normalize epoch lengths (pad/crop to uniform length)
    lengths = [ep.shape[1] for ep in aligned_data]
    max_len = max(lengths)

    padded_data = []
    for i, ep in enumerate(aligned_data):
        if ep.shape[1] != max_len:
            if debug:
                print(f"Epoch {i} adjusted from {ep.shape[1]} to {max_len} samples")
            ep = pad_or_crop_epoch(ep, max_len)
        padded_data.append(ep)

    aligned_data = padded_data

    # Step 3: Create MNE EpochsArray
    # We reuse the original sample positions from epochs to define event onsets
    events = np.array([
        [ep['start_sample'], 0, 1] for ep in epochs
    ])

    aligned_epochs = mne.EpochsArray(
        np.stack(aligned_data),
        info=raw.info,
        tmin=-pre_crossing_sec,
        events=events,
        event_id={"crossing": 1}
    )

    if debug:
        # Optional debug: visualize alignment on one example channel
        ref_ch = reference_crossing[0]  # channel 0
        test_ch = epochs[0]['data'][0, epochs[0]['crossing_start']:epochs[0]['crossing_end']]
        plt.plot(test_ch, label='Original Crossing')
        plt.plot(aligned_data[0][0, epochs[0]['crossing_start']:epochs[0]['crossing_start']+reference_duration],
                 label='DTW-aligned Crossing')
        plt.legend()
        plt.title("Channel 0: Original vs Aligned Crossing")
        plt.show()

    return aligned_epochs

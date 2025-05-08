from config import pre_crossing_sec, post_crossing_sec
import mne
import numpy as np
from autoreject import AutoReject


def pad_or_crop_epoch(epoch_data, target_length):
    """
    Pads or crops an EEG epoch to match a target number of timepoints (samples).
    
    Parameters:
        epoch_data (ndarray): EEG data for one epoch (n_channels, n_samples).
        target_length (int): Desired number of samples in time dimension.

    Returns:
        ndarray: Epoch with consistent shape (n_channels, target_length).
    """
    n_channels, current_length = epoch_data.shape  # Get dimensions of current epoch

    if current_length == target_length:
        # Already the correct length — no modification needed
        return epoch_data
    elif current_length < target_length:
        # Too short — pad with zeros at the end
        pad_width = target_length - current_length
        return np.pad(epoch_data, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Too long — truncate extra timepoints
        return epoch_data[:, :target_length]


def autoreject(raw, crossing_epochs):
    """
    Applies AutoReject to detect and interpolate bad EEG channels
    and discard bad epochs.

    Parameters:
        raw (mne.io.Raw): The original continuous EEG recording.
        crossing_epochs (list of dicts): Precomputed epoch windows with:
            - 'start_sample': Start index in raw data
            - 'end_sample': End index in raw data
            - 'data': (optional) original extracted EEG array
            - 'trial': list of annotated events for the trial

    Returns:
        epochs_clean (mne.EpochsArray): Cleaned, valid epochs (bad channels interpolated).
        kept_indices (list of int): Index positions of epochs retained (not rejected).
    """

    data = []        # Final list of uniformly-shaped EEG arrays for each epoch
    events = []      # Synthetic MNE-style event list: one event per epoch
    raw_epochs = []  # Temporary store of raw epoch EEG data before resizing

    max_len = 0      # Track the longest epoch length to pad others later

    # ----------------------------------------
    # Pass 1: Extract epochs from raw and track max length
    # ----------------------------------------
    for i, ep in enumerate(crossing_epochs):
        # Slice out EEG data for the trial using stored start/end indices
        epoch_data, _ = raw[:, ep['start_sample']:ep['end_sample']]
        
        raw_epochs.append(epoch_data)  # Keep this to resize in next step

        # Update maximum length if current epoch is longer than all previous ones
        if epoch_data.shape[1] > max_len:
            max_len = epoch_data.shape[1]

        # Create an artificial event at the start of this epoch
        # Format: [sample_index, placeholder, event_id]
        events.append([ep['start_sample'], 0, 1])

    # ----------------------------------------
    # Pass 2: Pad or crop all epochs to match max length
    # ----------------------------------------
    for ep_data in raw_epochs:
        # Ensure all epochs have exactly the same number of samples (max_len)
        fixed_data = pad_or_crop_epoch(ep_data, max_len)
        data.append(fixed_data)

    # Stack into a 3D NumPy array: shape = (n_epochs, n_channels, n_samples)
    data = np.stack(data)

    # Convert synthetic event list to NumPy array (required by MNE)
    events = np.array(events)

    # ----------------------------------------
    # Create MNE EpochsArray object from the data
    # ----------------------------------------
    epochs = mne.EpochsArray(
        data,             # EEG data with uniform shape
        info=raw.info,    # Copy metadata from raw recording (channel names, sfreq, etc.)
        events=events,    # Event list — one per epoch
        tmin=-pre_crossing_sec,  # Start of each epoch in seconds relative to event
        event_id={"crossing": 1} # Label all epochs as 'crossing'
    )

    # ----------------------------------------
    # Apply AutoReject
    # ----------------------------------------
    ar = AutoReject(
        n_interpolate=[1, 2, 3, 4],   # Try interpolating up to 4 bad channels
        consensus=[0.3, 0.7, 1.0],    # Different thresholds for what counts as a "bad" channel
        random_state=42              # Seed for reproducibility
    )

    # Fit the model and clean the epochs
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    # Identify indices of epochs that were *not* completely rejected
    kept_indices = np.where(~reject_log.bad_epochs)[0]

    # Return both the cleaned data and the indices for further filtering
    return epochs_clean, kept_indices

# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:30:54 2025

@author: danis
"""

import numpy as np
import pandas as pd
import mne
from scipy import signal
from tslearn.metrics import dtw_path
from config import debug, pre_crossing_sec, post_crossing_sec
from scripts import logger
import matplotlib.pyplot as plt  # Needed for debug plotting
from collections import Counter
import sys


def pad_or_crop_epoch(epoch_data, target_length):
    """
    Pad with zeros or crop an epoch to make it the same length as target_length.

    Parameters:
        epoch_data (ndarray): EEG data of shape (n_channels, n_samples)
        target_length (int): Desired number of time points (samples)

    Returns:
        ndarray: Padded or cropped EEG epoch with shape (n_channels, target_length)
    """
    n_channels, current_length = epoch_data.shape  # extracts the number of EEG channels and time samples from the epoch
    if current_length == target_length:  # if epoch is already correct lenght, return it without changing
        return epoch_data
    elif current_length < target_length:
        # f the epoch is too short, calculate how many extra samples are needed to match the target
        pad_width = target_length - current_length
        # Pad the end of the epoch with zeros (i.e., silence).
        return np.pad(epoch_data, ((0, 0), (0, pad_width)), mode='constant')
    else:  # Crop
        # If the epoch is longer than needed, cut off the extra samples at the end
        return epoch_data[:, :target_length]


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

    # empty list to store epochs,this will also include where obstacle crossing happened in the epoch
    crossing_epochs = []
    event_list = []

    def trial_is_absent(index):
        # if index + 1 >= len(meta):  # check if next row is within bounds of dataframe
        # If this is the last trial and +1 is out of bounds log a warning
        # logger.warning(
        # f"WARNING: Row {index + 1} does not exist in metadata (length={len(meta)}). Assuming obstacle is present.")
        # return False  # fallback assumption: obstacle is present
        return meta.iloc[index + 1]['ExistanceLevel'] == 'Absent'

    # ----------------------------
    # Step 1: Extract Epochs
    # This step takes continuous EEG data and makes epochs centered around when the participant crosses the obstacle. These segments will later be time-warped to match a reference.
    # Search each trial for the crossing start/end times (cs_time, ce_time)
    # Use pre_crossing_sec and +1.5 s to define the window
    # Pull out EEG data from the raw file for that segment
    # Calculate relative offsets to mark where the crossing happens inside the window
    # extracting eeg fram raw, reading event annotations, epoching-use samples; matching unity csv events and plotting brain response over time-use seconds
    # reference['data'] is a 2D NumPy array of shape (n_channels, n_samples). [:, crossing_start:crossing_end] means keeps all eeg channels (:) and slice timepoints corresponding to crossing
    # samples=seconds *sfreq
    # ce and cs time are in reference to full eeg recording, so when crossing starts and ends in full eeg recording (in seconds)
    # cs_offset and ce_offset-index where crossing starts and ends inside the epoch
    # ----------------------------
    # loop over each trial which is a list of event dictionaries that mark time-stamped events during that trial (obstaclecrossing start, end etc)

    for index, trial in enumerate(trialEvents):
        # these will hold start and end times of obstacle crossing, they're none now so i can check if they're found later
        cs_time, ce_time = None, None

        # loop over all events in the trial, Each event has a label (like ObstacleCrossingMid start) and a sample number which is converted to seconds
        for event in trial:
            label = event['label']
            # finding event time in seconds here
            time_in_sec = event['sample'] / sfreq

            # extract the correct labels based on obstacle condition. If the obstacle was absent, the crossing labels used are ObstacleCrossingMid start/end
            if trial_is_absent(index):
                if "ObstacleCrossingMid start" in label:
                    cs_time = time_in_sec
                elif "ObstacleCrossingMid end" in label:
                    ce_time = time_in_sec
            else:  # obstacle is present
                if "ObstacleCrossingBounds start" in label:  # these are the labels to extract when obstacle is present
                    cs_time = time_in_sec  # cs_time = timestamp when participant started crossing
                elif "ObstacleCrossingBounds end" in label:
                    ce_time = time_in_sec  # timestamp when participant finished crossing

        if cs_time is not None and ce_time is not None:  # define epoch time window
            # extracting data starting 2.5s before crossing, converted to sample indices
            start_sample = int((cs_time - pre_crossing_sec) * sfreq)
            # extracting data ending 1.5s after crossing, converted to sample indices
            end_sample = int((ce_time + post_crossing_sec) * sfreq)

            # slice the EEG raw signal for all channels in that time range
            # :=all eeg channels, so this will get all channels in the epoch
            data, _ = raw[:, start_sample:end_sample]
            # these offsets define where in the epoch the crossing occurs, cs_offset is the Index in the sliced data where crossing starts (always 2.5sec after epoch start
            cs_offset = int(pre_crossing_sec * sfreq)  # 1250
            # Index where crossing ends (length of crossing + pre time)
            # How much time has passed from the start of the epoch (i.e., 2.5 sec before the crossing) to the point when the crossing ends. and then get the sample
            ce_offset = int((ce_time - cs_time + pre_crossing_sec) * sfreq)

            crossing_epochs.append({  # save the eeg data for the trial and sample indices within the data for start/end of crossing
                'data': data,  # data is the epoch
                'crossing_start': cs_offset,  # sample index of when crossing starts in epoch
                'crossing_end': ce_offset  # sample index of when crossing ends in epoch
            })

            # prepare an event list in MNE format: [sample, 0, event_id] where event_id = 1 just labels all epochs the same, int(cs_time * sfreq) - Converts the crossing start time into a sample index in the full EEG recordin
            event_list.append([int(cs_time * sfreq), 0, 1])

    if not crossing_epochs:
        # if there are no trials with a valid crossing start and end, raise an error, maybe event labels or Unity metadata are off
        raise ValueError("No valid crossing epochs found")

    # ----------------------------
    # Step 2: Choose a Reference Crossing ((samples=sec*sfreq) this step selects one trial as the template to which all other obstacle crossing segments will be time-aligned using dtw))
    # ----------------------------
    median_idx = np.argsort([  # np.argsort(...) returns the indices that would sort the list of crossing durations from smallest to largest
        ep['crossing_end'] - ep['crossing_start']  # for each epoch compute how many samples long the actual crossing segment is. crossing_start and crossing_end were saved in Step 1 and define the section of the data where the participant was crossing the obstacle. this gives a list of crossing durations in sample
        for ep in crossing_epochs
    ])[len(crossing_epochs) // 2]  # gives the index of the middle item in the sorted list- the median trial. // operator rounds down (integer division) and python only keeps whole number part

    # This picks the median-length crossing trial from list of epochs.
    reference = crossing_epochs[median_idx]
    reference_crossing = reference['data'][:,  # reference['data'] is the full EEG epoch for that trial (including pre- and post-crossing parts)
                                           reference['crossing_start']:reference['crossing_end']]  # slices out just the crossing portion across all channels. reference_crossing becomes template waveform for DTW alignmen
    # stores the number of samples in the reference crossing. this will later be used to resample all DTW-aligned signals in other trials so they match this fixed length
    reference_duration = reference_crossing.shape[1]

    # ----------------------------
    # Step 3: Align Each Epoch to Reference
    # This makes sure that the crossing segments from all trials are the same length and aligned in time, even if each participant took a different amount of time to cross.
    # crossing_data.shape = (n_channels, n_samples_in_crossing), e.g.,= 2d array (32,200), each row is a channel and each column is a timepoint
    # ----------------------------
    aligned_data = []  # this list will be filled with aligned full epochs

    for epoch in crossing_epochs:  # processing each trial one by one
        # epoch before obstacle crossing
        pre_data = epoch['data'][:, :epoch['crossing_start']]
        crossing_data = epoch['data'][:,
                                      epoch['crossing_start']:epoch['crossing_end']]  # eeg segment during obstacle crossing
        # epoch segment after obstacle crossing
        post_data = epoch['data'][:, epoch['crossing_end']:]

        aligned_crossing = np.zeros(  # prepare an empty array to hold the aligned crossings
            (crossing_data.shape[0], reference_duration))  # crossing_data.shape[0] = number of channels. reference_duration = number of time points in the reference crossing
# align each channel individually now
        # loop over each eeg channel
        # crossing_data is shaped like (n_channels, n_samples), crossing_data.shape[0] gives the number of EEG channels, each loop handles the crossing segment for just one channel
        for ch_idx in range(crossing_data.shape[0]):
            # ref = the reference crossing for this channel
            # reference_crossing is median trial and contains eeg data during crossing from all channels. For each other trial i want to warp its crossing segment so that it matches the shape of the reference crossing segment. [ch_idx] picks out the crossing signal from just one EEG channel so array shape example- (200,), [:, np.newaxis] reshapes it to a column so shape: (200, 1) so 200 rows and one column, this shape is what dtw expects
            ref = reference_crossing[ch_idx][:, np.newaxis]
            # the current trial's crossing for the same channel
            # crossing_data[ch_idx]-gets just one eeg channle from current trial's crossing segment so shape becomes 1d array(200,), [:, np.newaxis] reshapes 1d array to 2D column vector (200, 1).This is needed because dtw_path() expects both signals to have 2 dimensions
            target = crossing_data[ch_idx][:, np.newaxis]

            # this finds the best alignment path between the reference and current trial crossing.tells how to warp the target to match the reference
            path, _ = dtw_path(ref, target)

            # this applies the DTW path to rearrange/stretch target so it aligns with ref.
            # makes a new array of the warped signal for just one eeg channel
            warped = np.array([target[j][0] for i, j in path])
            # resample warped data sp it so that it has exactly the same number of samples as the reference. this is still only 1 eeg channel.signal.resample uses Fourier-based interpolation
            warped_resampled = signal.resample(warped, reference_duration)

            # insert this channel’s aligned crossing segment into the empty array from earlier.
            # aligned_crossing holds all channels’ aligned segments,aligned_crossing[ch_idx] is the current channel’s row. the previous loop went over each eeg channel one by one and aligned each channel's crossing segment using dtw and now put them all in correct row of aligned_crossing
            aligned_crossing[ch_idx] = warped_resampled

        # combine original precrossing data, aligned data and post crossing data
        # np.hstack means horizontal stack and joins arrays along the second axis,
        # for eeg data thats the time axis (Samples), aligned_epoch has full aligned epochs for all channels, but it’s only for one trial at a time -the one currently being processed in the loop
        aligned_epoch = np.hstack((pre_data, aligned_crossing, post_data))
        # save data to aligned_epoch, this includes all epochs now
        aligned_data.append(aligned_epoch)

    # ----------------------------
    # Step 4: Create MNE Epochs
    # ----------------------------
    events = np.array(event_list)  # made earlier
    info = raw.info  # copies metadata from original raw EEG recording

    if debug:  # This plots the original vs DTW-aligned signal from channel 0

        # Plot one example comparison
        plt.plot(crossing_data[0], label='Original')
        plt.plot(aligned_crossing[0], label='DTW Aligned')
        plt.legend()
        plt.title("Channel 0: Original vs DTW-Aligned Crossing")
        plt.show()

    # Check that all epochs have the same length
    # This collects the number of timepoints (samples along the time axis) for each epoch into a list
    lengths = [ep.shape[1] for ep in aligned_data]
    print("Epoch length counts:", Counter(lengths))

    # Get the maximum length found
    max_len = max(lengths)

    # Pad any epochs that are shorter than the max
    padded_data = []  # a new list to hold the final uniform-length epochs
    for i, ep in enumerate(aligned_data):  # loop thorugh each epoch
        # If this epoch is shorter than the longest, it needs to be padded.
        if ep.shape[1] < max_len:
            if debug:
                print(
                    f"WARNING: Padding epoch {i} from {ep.shape[1]} to {max_len} samples.")  # print warnign that epoch has paddign applied
            # Pads the short epoch to the correct length using pad_or_crop_epoch() (adds zeros at the end)
            ep = pad_or_crop_epoch(ep, max_len)
        elif ep.shape[1] > max_len:
            if debug:
                print(
                    f"WARNING: Truncating epoch {i} from {ep.shape[1]} to {max_len} samples.")
            ep = pad_or_crop_epoch(ep, max_len)
        padded_data.append(ep)  # Adds the (possibly padded) epoch to the list

    # Replace aligned_data with padded version
    aligned_data = padded_data

    aligned_epochs = mne.EpochsArray(
        # Ensures uniform shape, incldues final EEG data: all trials aligned and combined, shape -(n_trials, n_channels, n_samples)
        np.stack(aligned_data),
        info=info,
        tmin=-pre_crossing_sec,
        events=events,
        event_id={"crossing": 1}
    )
    print("Channel 0 label:", raw.ch_names[0])

    return aligned_epochs

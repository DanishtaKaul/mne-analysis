# scripts/time_warping.py

import numpy as np
import pandas as pd
import mne
from scipy import signal

from scripts import logger


def align_epochs_with_dtw(raw, trialEvents, meta_info_path):
    """
    Create time-aligned epochs using Dynamic Time Warping (DTW).
    
    Parameters:
      - raw: mne.io.Raw object
      - trialEvents: List of trials, each containing event dictionaries
      - meta_info_path: Path to the metadata CSV file
      
    Returns:
      - aligned_epochs: mne.Epochs object with time-aligned data
    """
    sfreq = raw.info['sfreq']
    meta = pd.read_csv(meta_info_path)
    
    # Extract crossing events
    crossing_epochs = []
    event_list = []
    
    def trial_is_absent(index):
        return meta.loc[index + 1, 'ExistanceLevel'] == 'Absent'
    
    for index, trial in enumerate(trialEvents):
        cs_time = None
        ce_time = None
        
        for event in trial:
            label = event['label']
            time_in_sec = event['sample'] / sfreq
            
            if trial_is_absent(index):
                # For absent trials, look for midpoint crossing events
                if "ObstacleCrossingMid start" in label:
                    cs_time = time_in_sec
                elif "ObstacleCrossingMid end" in label:
                    ce_time = time_in_sec
            else:
                # For trials with obstacle, use crossing bounds events
                if "ObstacleCrossingBounds start" in label:
                    cs_time = time_in_sec
                elif "ObstacleCrossingBounds end" in label:
                    ce_time = time_in_sec
        
        # Only create an epoch if both crossing times were found
        if cs_time is not None and ce_time is not None:
            # Create epoch around crossing period with pre and post padding
            start_sample = int((cs_time - 2.5) * sfreq)
            end_sample = int((ce_time + 1.5) * sfreq)
            
            # Extract data for this epoch
            data, times = raw[:, start_sample:end_sample]
            
            # Mark the crossing start and end points within the epoch
            cs_offset = int(2.5 * sfreq)  # 2.5s pre-crossing
            ce_offset = int((ce_time - cs_time + 2.5) * sfreq)  # crossing end relative to epoch start
            
            # Store the epoch data and crossing points
            crossing_epochs.append({
                'data': data,
                'crossing_start': cs_offset,
                'crossing_end': ce_offset,
                'total_length': end_sample - start_sample
            })
            
            # Store event for later creation of MNE Epochs object
            event_list.append([int(cs_time * sfreq), 0, 1])
    
    if not crossing_epochs:
        raise ValueError("No valid crossing epochs found")
    
    # Calculate median crossing duration for reference
    crossing_durations = [ep['crossing_end'] - ep['crossing_start'] for ep in crossing_epochs]
    median_duration = int(np.median(crossing_durations))
    
    # Use DTW to align all crossing periods to a reference
    aligned_data = []
    tmin = -2.5
    
    for epoch in crossing_epochs:
        # Extract pre-crossing, crossing, and post-crossing segments
        pre_data = epoch['data'][:, :epoch['crossing_start']]
        crossing_data = epoch['data'][:, epoch['crossing_start']:epoch['crossing_end']]
        post_data = epoch['data'][:, epoch['crossing_end']:]
        
        # Apply DTW to the crossing segment (channel by channel)
        aligned_crossing = np.zeros((crossing_data.shape[0], median_duration))
        
        for ch_idx in range(crossing_data.shape[0]):
            # Resample crossing data to the median duration
            # This is a simplified approach - for true DTW you'd use the tslearn library
            aligned_crossing[ch_idx] = signal.resample(crossing_data[ch_idx], median_duration)
        
        # Keep pre and post segments unchanged
        aligned_epoch = np.hstack((pre_data, aligned_crossing, post_data))
        aligned_data.append(aligned_epoch)
    
    # Create MNE Epochs object with aligned data
    events = np.array(event_list)
    info = raw.info
    
    # Create epochs with aligned data
    aligned_epochs = mne.EpochsArray(
        np.array(aligned_data),
        info,
        tmin=tmin,
        events=events,
        event_id={1: 1}
    )
    
    return aligned_epochs

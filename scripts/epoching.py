import pandas as pd 

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

import pandas as pd
from config import (pre_crossing_sec, post_crossing_sec, debug)
import sys

# helper function to check if the current trial is absent
def trial_is_absent(meta, index):
        return meta.iloc[index]['ExistanceLevel'] == 'Absent'
        # if index + 1 >= len(meta):  # check if next row is within bounds of dataframe
        # If this is the last trial and +1 is out of bounds log a warning
        # logger.warning(
        # f"WARNING: Row {index + 1} does not exist in metadata (length={len(meta)}). Assuming obstacle is present.")
        # return meta_row['ExistanceLevel'] == 'Absent'

def identify_epochs(raw, trialEvents, meta_info_path):

    """    
    ----------------------------
    Step 1: Extract Epochs
    This step takes continuous EEG data and makes epochs centered around when the participant crosses the obstacle. These segments will later be time-warped to match a reference.
    Search each trial for the crossing start/end times (cs_time, ce_time)
    Use pre_crossing_sec and +1.5 s to define the window
    Pull out EEG data from the raw file for that segment
    Calculate relative offsets to mark where the crossing happens inside the window
    extracting eeg fram raw, reading event annotations, epoching-use samples; matching unity csv events and plotting brain response over time-use seconds
    reference['data'] is a 2D NumPy array of shape (n_channels, n_samples). [:, crossing_start:crossing_end] means keeps all eeg channels (:) and slice timepoints corresponding to crossing
    samples=seconds *sfreq
    ce and cs time are in reference to full eeg recording, so when crossing starts and ends in full eeg recording (in seconds)
    cs_offset and ce_offset-index where crossing starts and ends inside the epoch
    ----------------------------
    loop over each trial which is a list of event dictionaries that mark time-stamped events during that trial (obstaclecrossing start, end etc)


    empty list to store epochs,this will also include where obstacle crossing happened in the epoch
    """
    
    sfreq = raw.info['sfreq']
    meta = pd.read_csv(meta_info_path)
    
    crossing_epochs = []
    event_list = []
    
    for idx, (trial, meta_row) in enumerate(zip(trialEvents, meta)):
        # these will hold start and end times of obstacle crossing, they're none now so i can check if they're found later
        cs_time, ce_time = None, None

        # loop over all events in the trial, Each event has a label (like ObstacleCrossingMid start) and a sample number which is converted to seconds
        for event in trial:
            label = event['label']
            # finding event time in seconds here
            time_in_sec = event['sample'] / sfreq

            # extract the correct labels based on obstacle condition. If the obstacle was absent, the crossing labels used are ObstacleCrossingMid start/end
            if trial_is_absent(meta, idx):
                if "Toe" and "Mid" and "start" in label and cs_time is None:
                    cs_time = time_in_sec
                elif "Heel" and "Mid" and "end" in label and ce_time is None:
                    ce_time = time_in_sec
            else:  # obstacle is present
                # these are the labels to extract when obstacle is present
                if "Toe" and "Bounds" and "start" in label and cs_time is None:
                    cs_time = time_in_sec  # cs_time = timestamp when participant started crossing
                elif "Heel" and "Bounds" and "end" in label and ce_time is None:
                    ce_time = time_in_sec  # timestamp when participant finished crossing

        if debug and cs_time is None or ce_time is None:
            sys.exit(
                f"Marker missing, a crossing time is not found. \n Crossing start time: {cs_time} \n Crossing end time: {ce_time} \n Event: {event}")

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
                'crossing_end': ce_offset,  # sample index of when crossing ends in epoch
                'start_sample': start_sample,
                'end_sample': end_sample,
                'trial': trial,
            })

            # prepare an event list in MNE format: [sample, 0, event_id] where event_id = 1 just labels all epochs the same, int(cs_time * sfreq) - Converts the crossing start time into a sample index in the full EEG recordin
            event_list.append([int(cs_time * sfreq), 0, 1])

    if not crossing_epochs:
        # if there are no trials with a valid crossing start and end, raise an error, maybe event labels or Unity metadata are off
        raise ValueError("No valid crossing epochs found")

    return crossing_epochs

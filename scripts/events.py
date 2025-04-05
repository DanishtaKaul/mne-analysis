import mne

"""
    Create trial events using the absolute sample indices from the events array.
    Each event is stored as a dictionary with 'sample' (absolute sample index)
    and 'label' (annotation label).

    @Params raw
     The raw data file
"""

def create_trial_events(raw):
   
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


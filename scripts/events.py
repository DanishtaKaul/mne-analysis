import mne
import sys
from scripts import logger

"""
    Create trial events using the absolute sample indices from the events array.
    Each event is stored as a dictionary with 'sample' (absolute sample index)
    and 'label' (annotation label).


    Essentially, this function gives an array of sequential events (array) with the event sample and the event label (object)
    Trial event example:
    {"sample": 7446, "label": "Return Walk at 149.4744"}

    @Params raw
     The raw data file
"""


def create_trial_events(raw):
    events, event_id = mne.events_from_annotations(raw)
    code_to_label = {code: label for label, code in event_id.items()}

    trial_events = []
    current_trial = []

    for event in events:
        sample = event[0]
        code = event[2]
        # label = code_to_label.get(code, "")
        label = code_to_label.get(code, "")
        ev_dict = {"sample": sample, "label": label}

        if "Start Trial" in label:
            if current_trial:
                trial_events.append(current_trial)
            current_trial = [ev_dict]
        else:
            current_trial.append(ev_dict)

    if current_trial:
        trial_events.append(current_trial)

    # REMOVE arrays with only one item aka: '['Start Trial']'
    trial_events = [trial for trial in trial_events if len(trial) > 1]

    return trial_events

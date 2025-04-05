import mne
import numpy as np
from scipy import signal

from scripts import logger

"""
 Apply bandpass (0-40Hz) and detrend data

 @Params raw
     The raw data file
"""

def filter_and_detrend_data(raw):
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
    signal.detrend(raw.get_data(), axis=1)



"""
 Apply bandpass and detrend data

 @Params raw
     The raw data file
"""


def apply_ICA(raw):

    ica = mne.preprocessing.ICA(
        n_components=29, random_state=42, method='fastica')
    ica.fit(raw, decim=3)

    # Start with no components excluded
    ica.exclude = []

    # Parameters for iterative detection, z thresh-ica components are compared to signals from fp1, fp2, f7, f8 and if the z score of correlation with these channels is more than 3.5 they are considered artifact
    max_ic = 8       # Maximum number of EOG components expected
    z_thresh = 3.5   # Initial high z-score threshold
    z_step = 0.05    # Step to decrease the threshold iteratively

    # While the max number of EOG components hasn't been reached, keep lowering threshold
    while len(ica.exclude) < max_ic:
        eog_inds, eog_scores = ica.find_bads_eog(
            raw,
            # Using frontal channels as EOG proxies
            ch_name=['FP1', 'FP2', 'F7', 'F8'],
            threshold=z_thresh
        )

        # Combine new findings with any already excluded
        ica.exclude = list(set(ica.exclude + eog_inds))

        print(f"Threshold: {z_thresh:.2f}, EOG components found: {eog_inds}")

        # Lower threshold for next iteration if still haven't reached max_ic
        z_thresh -= z_step

        # If no new components were found in this iteration, break early
        if len(eog_inds) == 0:
            break

    print(f"Final excluded EOG components: {ica.exclude}")

    # Remove (exclude) the identified EOG components from raw data
    ica.apply(raw)

    # Visual checks
    ica.plot_scores(eog_scores)
    ica.plot_components()

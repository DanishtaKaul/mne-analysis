# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:27:08 2025

@author: danis
"""

import mne
import numpy as np
from scipy import signal
from scipy.stats import kurtosis
from mne_icalabel import label_components
from mne.filter import detrend


# z-score -How far is a value from the average, measured in standard deviations?
# mean correlation is the average correlation between each ICA component and the eye channels (FP1, FP2, F7, F8)
# Steps happening during ica-For each ICA component, compute the Pearson correlation with each eye channel (FP1, FP2, F7, F8).
# For each component, keep the highest absolute correlation value across these channels.
# Calculate the mean and standard deviation of these max correlation values across all components.
# For each component, compute its z-score: (max_corr - mean) / std.
# If a component's z-score is above the threshold (e.g., 3.5), flag it as an artifact (likely eye movement).
# SNR, sources[:, signal_window]	Select signal time points (after 2 seconds) for each ICA component..var(axis=1)	Calculate the variance of the signal for each ICA component.sources[:, baseline_window].var(axis=1)	Calculate the variance of the baseline (standing still) for each ICA component.+ 1e-12	Add a very tiny number to baseline variance to avoid dividing by zero (if variance accidentally = 0).
# High SNR means signal is much stronger than baseline - good brain signal.


def filter_and_detrend_data(raw):
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
    detrended_data = detrend(raw.get_data(), order=1, axis=1)
    raw._data = detrended_data  # avoid recreating Raw object
    raw.set_eeg_reference('average', projection=False)
    # Plot Power Spectral Density
    raw.plot_psd(fmax=60)  # check for line noise
    return raw


def apply_ICA(raw):
    # Extended Infomax ICA
    ica = mne.preprocessing.ICA(
        n_components=29, random_state=42, method='infomax', fit_params=dict(extended=True)
    )
    # # random_state- fixed number 42 will alway give same result each time code is run on same data
    ica.fit(raw, decim=3)
# decim=3 means that ICA uses every 3rd time point to learn what artifacts look like then removes those artifacts from full original EEG data

    # 1 ICLabel # finds ICA components that are muscle artifacts, eye blinks/movements, heartbeat noise, line noise, broken channels, or strange noise
    labels = label_components(raw, ica, method='iclabel')
    print("ICLabel output keys:", labels.keys())

    artifact_inds = set(
        idx for idx, label in enumerate(labels['labels'])
        if label != 'brain' and labels['y_pred_proba'][idx].max() > 0.8
    )
    print("ICLabel labels:", labels['labels'], flush=True)
    print("\nIC component → ICLabel mapping (with confidence):", flush=True)
    for idx, (label, probs) in enumerate(zip(labels['labels'], labels['y_pred_proba'])):
        confidence = round(float(np.max(probs)), 3)
        print(
            f"Component {idx}: {label} (confidence = {confidence})", flush=True)

    # 3 EOG correlation (dynamic z-score thresholding)
    print("\n=== EOG Correlation Check (Dynamic Threshold) ===")

    z_thresh = 3.5     # Start with a high z-score threshold for strict EOG detection
    # Gradually reduce the threshold by this amount if not enough components are found
    z_step = 0.05
    max_ic = 2         # minimum number of EOG components to detect
    num_excl = 0       # Number of components flagged so far

    # Iteratively lower the threshold until at least max_ic components are found or threshold gets too low
    while num_excl < max_ic and z_thresh > 0:
        # Find ICA components that correlate with EOG channels above the current z-score threshold
        eog_inds, eog_scores = ica.find_bads_eog(
            raw, ch_name=['FP1', 'FP2', 'F7', 'F8'], threshold=z_thresh
        )

        num_excl = len(eog_inds)  # Update the count of flagged components

        if num_excl < max_ic:
            # Not enough components detected — try a more lenient threshold
            z_thresh -= z_step

    # Convert to set for easy combination later (with other artifact methods)
    eog_inds = set(eog_inds)

    # Print the components found and the final threshold used
    print(
        f"EOG correlation (dynamic z > {round(z_thresh, 2)}): {sorted(eog_inds)}")
    # Now that ICA is fitted, extract its data for further checks

    # ICA time series (sources): each row is one component, sources contains timecourse of each component over time
    sources = ica.get_sources(raw).get_data()

    # ICA spatial maps: how much each component contributes to each channel or where on the scalp each ICA component is strongest, used in kurtosis
    maps = ica.get_components().T

    # 4 Autocorrelation (threshold: z < -2) How similar is a signal to itself if its shifted a little bit in time, low autocorrelation could be muscle artifact. The autocorrelation method identifies noisy components by flagging those with low self-correlation over a 20 ms lag, using a cutoff of 2 standard deviations below the mean.

    def get_autocorr(ic_sources, lag_samples):
        autocorrs = []
        for ic in ic_sources:
            # Calculates the Pearson correlation between the signal and itself, shifted by 20 samples (lag = 20 samples), low correlation-brain activity, high correlation-muscle artifact
            ac = np.corrcoef(ic[:-lag_samples], ic[lag_samples:])[0, 1]
            # save autocorrelation value for each component
            autocorrs.append(ac)
        # After looping over all components, returns the array of autocorrelation values.
        return np.array(autocorrs)
    # Use sampling frequency to calculate 20 ms lag in samples
    sfreq = raw.info['sfreq']  # your EEG data's sampling rate, e.g. 500 Hz
    lag_samples = int(0.020 * sfreq)  # convert 20 milliseconds to samples

    # calculate autocorrelation for every ICA component.
    autocorrs = get_autocorr(sources, lag_samples=lag_samples)
    # calculate z score of each autocorrelation value. A component that has very low autocorrelation compared to most others will have a strong negative z-score,Components close to average will have z-scores around 0.
    z_autocorr = (autocorrs - autocorrs.mean()) / autocorrs.std()
    # checks for every ICA component whether its z-scored autocorrelation value is less than -2
    auto_inds = set(np.where(z_autocorr < -2)[0])
    print("\n=== Autocorrelation Check ===")
    for i in range(len(autocorrs)):
        print(
            f"Component {i:02d}: autocorr = {autocorrs[i]:.3f}, z = {z_autocorr[i]:+.2f}")
    print(f"Flagged (z < -2): {sorted(auto_inds)}")

    # 5 Focality (kurtosis; threshold: z > 2) kurtosis- how peaky a distribution of values is, is signal spread out smoothly over a few electrodes or sharply focused on 1-2 electrodes. detects ICA components with unusually high kurtosis by flagging those more than 2 standard deviations above the mean
    # Calculates the kurtosis for each ICA component’s spatial map (the pattern across electrodes), maps has ica topographies, kurt has one kurtosis value per ica component
    kurt = kurtosis(maps, axis=1)
    # Calculates the z-score of each kurtosis value to find which components are unusually focal (very high kurtosis) compared to the others.
    z_kurt = (kurt - kurt.mean()) / kurt.std()
    # Flags ICA components where the kurtosis z-score is greater than 2.
    focal_inds = set(np.where(z_kurt > 2)[0])
    print("\n=== Focality (Kurtosis) Check ===")
    for i in range(len(kurt)):
        print(
            f"Component {i:02d}: kurtosis = {kurt[i]:.2f}, z = {z_kurt[i]:+.2f}")
    print(f"Flagged (z > 2): {sorted(focal_inds)}")

    # 6. SNR (threshold: < 1)
    # 6. SNR (Updated for 2s baseline and 1.5s walking)

    # baseline_duration = 2.0  # they stand for 2 seconds
    # walking_duration = 1.5   # walking window
    # sfreq = raw.info['sfreq']

    # annotations = raw.annotations
    # descriptions = [desc.split(" at ")[0].strip()
    # for desc in annotations.description]
    # trial_starts = [onset for desc, onset in zip(
    # descriptions, annotations.onset) if "Start Trial" in desc]

    # snr_values = []

    # print("\n=== Trial Timing Summary (Adjusted for experiment) ===")

    # for i, start_onset in enumerate(trial_starts):
    # baseline_start = int(start_onset * sfreq)
    # baseline_end = int((start_onset + baseline_duration) * sfreq)
    # trial_end = trial_starts[i + 1] if i + \
    # 1 < len(trial_starts) else raw.times[-1]

    # walk_onset = None

    # for desc, onset in zip(descriptions, annotations.onset):
    # if start_onset < onset < trial_end:
    # if desc == "Return Walk":
    # walk_onset = onset + 1.0
    # break
    # elif "ObstacleCrossingBounds start" in desc and walk_onset is None:
    # walk_onset = onset - 1.5

    # if walk_onset is None:
    # print(f"Trial {i+1}: Skipped (no valid walking label)")
    # continue

    # walk_start = int(walk_onset * sfreq)
    # walk_end = int((walk_onset + walking_duration) * sfreq)

    # if walk_end > sources.shape[1] or baseline_end > sources.shape[1]:
    # print(f"Trial {i+1}: Skipped (window out of bounds)")
    # continue

    # baseline_var = sources[:, baseline_start:baseline_end].var(axis=1)
    # walk_var = sources[:, walk_start:walk_end].var(axis=1)
    # snr = walk_var / (baseline_var + 1e-12)
    # snr_values.append(snr)

    # print(f"Trial {i+1}:")
    # print(
    # f"  Baseline: {start_onset:.2f}s → {start_onset + baseline_duration:.2f}s")
    # print(
    # f"  Walking : {walk_onset:.2f}s → {walk_onset + walking_duration:.2f}s\n")

    # Final decision
    # if snr_values:
    # mean_snr = np.mean(snr_values, axis=0)

    # More tolerant threshold or consider z-scores
    # snr_inds = set(np.where(mean_snr < 0.75)[0])
    # print("SNR-based artifact components (SNR < 0.75):", sorted(snr_inds))
    # else:
    # mean_snr = np.zeros(sources.shape[0])
    # snr_inds = set()
    # print("No valid SNR windows found.")

    # 7 Correlation with any channel (threshold: z > 4). The Correlation with channel check flags ICA components that are highly correlated with a single EEG channel using a threshold of 4 sd
    raw_data = raw.get_data()  # raw_data is now shaped (n_channels, n_times).
    corrs = []  # Create an empty list called corrs to store correlation values
    for ic in sources:
        # corr_vals= list of correlation values between the ICA component and each raw EEG channel. np.corrcoef(ic, ch)[0, 1] = Pearson correlation coefficient between ICA component and one channel.
        corr_vals = [np.corrcoef(ic, ch)[0, 1] for ch in raw_data]
        # np.max(np.abs(corr_vals)) = take the maximum absolute correlation (strongest link, ignoring sign)
        corrs.append(np.max(np.abs(corr_vals)))
    # get a z-score for each ICA component's maximum correlation
    z_corr = (np.array(corrs) - np.mean(corrs)) / np.std(corrs)
    # Find the indices of ICA components where the z-scored correlation > 4, meaning the component’s maximum correlation is extremely high compared to other components
    corr_inds = set(np.where(z_corr > 4)[0])

    print("\n=== Artifact Detection Summary ===")
    print(f"ICLabel (non-brain, confidence > 0.8): {sorted(artifact_inds)}")
    print(f"EOG correlation (z > 3.5): {sorted(eog_inds)}")
    print(f"Low autocorrelation (z < -2): {sorted(auto_inds)}")
    print(f"High kurtosis (z > 2): {sorted(focal_inds)}")
    # print(f"Low SNR (< 1): {sorted(snr_inds)}")
    print(f"High correlation with EEG channels (z > 4): {sorted(corr_inds)}")

    # Combine all unique artifact indices
    all_excludes = (artifact_inds | eog_inds |
                    auto_inds | focal_inds | corr_inds)
    ica.exclude = list(all_excludes)
    print(f"Excluded ICA components (all methods): {sorted(ica.exclude)}")

    # Visual inspection
    ica.plot_components()

    # Plot properties of excluded components (topo + time series + spectrum)
    if ica.exclude:
        print(f"\nPlotting properties for excluded components: {ica.exclude}")
        import matplotlib.pyplot as plt
        plt.close('all')  # avoid memory overload from too many open plots

        ica.plot_properties(raw, picks=ica.exclude)

    # Apply ICA
    ica.apply(raw)

    return raw

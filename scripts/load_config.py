import re
from collections import defaultdict
import mne


def load_and_configure_data(file_path, montage_path):
    raw = mne.io.read_raw_fif(file_path, preload=True)
    raw.set_channel_types({ch_name: 'eeg' for ch_name in raw.info['ch_names']})
    raw.rename_channels({
        'EE214-000000-000116-02-DESKTOP-E45KF45_0': 'FP1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_1': 'FPz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_2': 'FP2',
        'EE214-000000-000116-02-DESKTOP-E45KF45_3': 'F7',
        'EE214-000000-000116-02-DESKTOP-E45KF45_4': 'F3',
        'EE214-000000-000116-02-DESKTOP-E45KF45_5': 'Fz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_6': 'F4',
        'EE214-000000-000116-02-DESKTOP-E45KF45_7': 'F8',
        'EE214-000000-000116-02-DESKTOP-E45KF45_8': 'FC5',
        'EE214-000000-000116-02-DESKTOP-E45KF45_9': 'FC1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_10': 'FC2',
        'EE214-000000-000116-02-DESKTOP-E45KF45_11': 'FC6',
        'EE214-000000-000116-02-DESKTOP-E45KF45_12': 'M1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_13': 'T7',
        'EE214-000000-000116-02-DESKTOP-E45KF45_14': 'C3',
        'EE214-000000-000116-02-DESKTOP-E45KF45_15': 'Cz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_16': 'C4',
        'EE214-000000-000116-02-DESKTOP-E45KF45_17': 'T8',
        'EE214-000000-000116-02-DESKTOP-E45KF45_18': 'M2',
        'EE214-000000-000116-02-DESKTOP-E45KF45_19': 'CP5',
        'EE214-000000-000116-02-DESKTOP-E45KF45_20': 'CP1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_21': 'CP2',
        'EE214-000000-000116-02-DESKTOP-E45KF45_22': 'CP6',
        'EE214-000000-000116-02-DESKTOP-E45KF45_23': 'P7',
        'EE214-000000-000116-02-DESKTOP-E45KF45_24': 'P3',
        'EE214-000000-000116-02-DESKTOP-E45KF45_25': 'Pz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_26': 'P4',
        'EE214-000000-000116-02-DESKTOP-E45KF45_27': 'P8',
        'EE214-000000-000116-02-DESKTOP-E45KF45_28': 'POz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_29': 'O1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_30': 'Oz',
        'EE214-000000-000116-02-DESKTOP-E45KF45_31': 'O2',
        'EE214-000000-000116-02-DESKTOP-E45KF45_32': 'N/A1',
        'EE214-000000-000116-02-DESKTOP-E45KF45_33': 'N/A2',
    })
    raw.drop_channels({'M1', 'M2', 'N/A1', 'N/A2'})
    montage = mne.channels.read_custom_montage(montage_path)
    raw.set_montage(montage)

# =============================================================================
#     print("\n=== Trigger Summary by Trial ===")
#     annotations = raw.annotations
#
#     def clean_trigger(label):
#         label = label.split(" at ")[0].strip()
#         label = re.sub(r": \d+$", "", label)
#         return label
#
#     trial_starts = [
#         (i, onset) for i, (desc, onset) in enumerate(zip(annotations.description, annotations.onset))
#         if "Start Trial" in desc
#     ]
#
#     trial_events = defaultdict(list)
#
#     for idx, (start_idx, start_onset) in enumerate(trial_starts):
#         end_onset = trial_starts[idx + 1][1] if idx + \
#             1 < len(trial_starts) else raw.times[-1]
#         for desc, onset in zip(annotations.description, annotations.onset):
#             if start_onset <= onset < end_onset:
#                 trial_events[idx].append(clean_trigger(desc))
#
#     for trial_idx, events in trial_events.items():
#         print(f"Trial {trial_idx + 1}: {sorted(set(events))}")
#
#     common_labels = set(trial_events[0])
#     for evs in trial_events.values():
#         common_labels &= set(evs)
#
#     print("\nTriggers present in EVERY trial:", sorted(common_labels))
#
#     print("\n=== Estimated Walking Onset Per Trial ===")
#
#     for idx, (start_idx, start_onset) in enumerate(trial_starts):
#         end_onset = trial_starts[idx + 1][1] if idx + \
#             1 < len(trial_starts) else raw.times[-1]
#
#         walk_onset = None
#         fallback_onset = None
#
#         for desc, onset in zip(annotations.description, annotations.onset):
#             if start_onset <= onset < end_onset:
#                 cleaned = clean_trigger(desc)
#
#                 if cleaned == "Return Walk":
#                     walk_onset = onset
#                     break
#                 elif "ObstacleCrossingBounds start" in cleaned and fallback_onset is None:
#                     fallback_onset = onset - 1.0
#
#         final_walk_onset = walk_onset if walk_onset else fallback_onset
#         if final_walk_onset:
#             print(
#                 f"Trial {idx + 1}: Walking estimated at {final_walk_onset:.2f} s")
#         else:
#             print(f"Trial {idx + 1}: Walking onset could not be estimated.")
# =============================================================================
    # Plot PSD before filtering
    raw.plot_psd(fmax=60)

    return raw

import mne

"""
 Reads raw file, renames electrode channels, sets the montage

 @Params file_path
     The filepath to the raw file

 @Params montage_path
     The filepath to the montage file
"""


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
    }
    )
    raw.drop_channels({'M1', 'M2', 'N/A1', 'N/A2'})
    montage = mne.channels.read_custom_montage(montage_path)
    raw.set_montage(montage)
    return raw


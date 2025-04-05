from scripts import logger
from scripts.load_config import load_and_configure_data
from scripts.preprocessing import filter_and_detrend_data, apply_ICA
from scripts.events import create_trial_events
from scripts.epoching import create_epochs
from scripts.time_warping import align_epochs_with_dtw
from config import montage_path

"""
    
"""
def process_block(file_path, meta_info_path, montage_path=montage_path):
    logger.info(f"Running preprocessing block on: {file_path}")

    raw = load_and_configure_data(file_path, montage_path)

    # filter_and_detrend_data(raw)
    # apply_ICA(raw)

    trial_events = create_trial_events(raw)

    epochs = create_epochs(raw, trial_events, meta_info_path)

    # aligned_epochs = align_epochs_with_dtw(raw, trial_events, meta_info_path)

    # logger.info(f"Created {len(epochs)} raw epochs")
    # logger.info(f"Aligned shape: {aligned_epochs.get_data().shape}")


    
import os

from scripts import logger  
from scripts.block_pipeline import process_block 
from config import (experiments, debug)

def process_multiple_experiments():
    for experiment in experiments:
        navigate_experiment(experiment)

# Navigate through an experiments file structure in the continuous matter of which the experiment ran
def navigate_experiment(experiment_root):
    eeg_dir = None
    unity_dir = None
    
    # Find the EEG and Unity directories.
    for item in os.listdir(experiment_root):
        item_path = os.path.join(experiment_root, item)
        if os.path.isdir(item_path):
            lower_item = item.lower()
            if 'fif' in lower_item:
                eeg_dir = item_path
            elif 'unity' in lower_item:
                unity_dir = item_path

    if eeg_dir is None:
        logger.warning("No EEG folder (with 'fif' in its name) found in experiment root.")
        return
    if unity_dir is None:
        logger.warning("No Unity folder (with 'unity' in its name) found in experiment root.")
        return

    # Define conditions.
    light_conditions = ['light', 'ambient', 'dark']
    obstacle_conditions = ['expected', 'unexpected']

    # Process each combination of conditions.
    for light in light_conditions:
        for obstacle in obstacle_conditions:

            # Search for the appropriate EEG file.
            eeg_file_path = None
            for file in os.listdir(eeg_dir):
                file_lower = file.lower()
                if file_lower.endswith('.fif') and (light in file_lower):
                    # For "expected", ensure "unexpected" is not present.
                    if obstacle == 'expected' and 'unexpected' in file_lower:
                        continue
                    # For "unexpected", ensure "unexpected" is present.
                    elif obstacle == 'unexpected' and 'unexpected' not in file_lower:
                        continue
                    eeg_file_path = os.path.join(eeg_dir, file)
                    break

            if eeg_file_path is None:
                logger.info(f"No EEG file found for condition: {light} & {obstacle}")
                continue

            # Search for the appropriate Unity block folder.
            unity_block_dir = None
            for folder in os.listdir(unity_dir):
                folder_lower = folder.lower()
                folder_path = os.path.join(unity_dir, folder)
                if os.path.isdir(folder_path) and (light in folder_lower):
                    if obstacle == 'expected' and 'unexpected' in folder_lower:
                        continue
                    elif obstacle == 'unexpected' and 'unexpected' not in folder_lower:
                        continue
                    unity_block_dir = folder_path
                    break

            if unity_block_dir is None:
                logger.info(
                    f"No Unity block folder found for condition: {light} & {obstacle}")
                continue

            # Locate the subfolder starting with 'S'.
            s_folder = None
            for folder in os.listdir(unity_block_dir):
                folder_path = os.path.join(unity_block_dir, folder)
                if os.path.isdir(folder_path) and folder.lower().startswith('s'):
                    s_folder = folder_path
                    break

            if s_folder is None:
                logger.info(
                    f"No subfolder starting with 'S' found in Unity block folder for condition: {light} & {obstacle}")
                continue

            # Check if trial_results.csv exists.
            trial_results_path = os.path.join(s_folder, "trial_results.csv")
            if not os.path.exists(trial_results_path):
                logger.info(
                    f"trial_results.csv not found in {s_folder} for condition: {light} & {obstacle}")
                continue

            # Process the block.
            if (debug):
                logger.info("\n ====== \n")
                logger.info(f"CURRENT EXPERIMENT: {experiment_root}")
                logger.info(
                    f"Processing condition: {light.title()}, {obstacle.title()}")
                logger.info(f"file_path: {eeg_file_path}")
                logger.info(f"meta_info_path: {trial_results_path}")
                logger.info("\n ====== \n")
            else:
                process_block(eeg_file_path, trial_results_path)

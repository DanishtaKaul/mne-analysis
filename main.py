# main.py

"""
EEG Preprocessing Pipeline

1. Load raw data
2. Rename channels and apply montage
3. Filter + detrend
4. Run ICA to remove EOG artifacts
5. Extract events and trials
6. Epoch data around obstacle crossings
7. Align epochs using DTW
8. Save / visualize output
"""


from scripts.navigation import navigate_experiment, process_multiple_experiments
from scripts.block_pipeline import process_block
from config import (
    file_path,
    meta_info_path,
    experiment_root,
    experiments,
    montage_path,
    debug,
)

def main():
    # process_multiple_experiments(experiments)
    # navigate_experiment(experiment)
    process_block(file_path, meta_info_path, montage_path)

# Lets us run directly from terminal
if __name__ == "__main__":
    main()

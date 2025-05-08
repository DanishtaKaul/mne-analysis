# main.py

"""
EEG Preprocessing Pipeline

1. Load raw data
2. Rename channels and apply montage
3. Filter + detrend
4. Run ICA to remove EOG artifacts and then use autoreject
5. Extract events and trials
6. Epoch data around obstacle crossings
7. Align epochs using DTW
8. Save / visualize output
"""


from scripts.navigation import navigate_experiment, process_multiple_experiments
from scripts.block_pipeline import process_block
from config import (
    experiments,
    montage_path,
    debug,
    file_path,
    meta_info_path
)


def main():
   #process_multiple_experiments()
  # navigate_experiment(experiment)
   #process_block(r"E:\PID 5\FIF\PID 5 DARK UNEXPECTED.fif",r"E:\PID 5\UNITY\PID 5 DARK UNEXPECTED\S002\trial_results.csv", montage_path)
   process_block(file_path, meta_info_path, montage_path)


# Lets it run directly from terminal
if __name__ == "__main__":
    main()

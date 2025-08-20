import os 
"""
Taken from README.md:
### Dataset Structure

Before starting model training using the command-line interface provided below, you must first configure your dataset dictionary file located at `training_code/ds_dict.py`.

This file defines a Python dictionary named `ds_paths`, where you should specify paths to the `train`, `val`, and `test` partitions of your dataset. Each partition should be a CSV file with the following three columns:

1. `wav_path` — Path to the WAV audio file.  
2. `tg_path` — Path to the corresponding `.TextGrid` file containing forced alignment.  
3. `raw_text` — Ground truth transcription.

> **Note:** The dictionary key (i.e., the name of the dataset) will be used by the training script to identify and load the dataset correctly.
"""

ds_paths = {
    'LIBRI-960-ALIGNED': {
        'train': f"{os.environ.get('HOME')}/path/to/datasets/libri_train_960_train.csv",  # used on training steps.
        'val': f"{os.environ.get('HOME')}/path/to/datasets/libri_train_960_val.csv",  # used on validation steps.
        'test': f"{os.environ.get('HOME')}/path/to/datasets/libri_train_960_test.csv"  # used on evaluation.
    },
    # Add you entries below.
}

# This file contains custom conditions for scrubbing data, meant to work with the
# provided example generation config. When customizing, make sure this read path is consistent
# with your prior generator_config's write path
# All rootfile paths are relative to the project's artifacts folder. 
# The annotations file name must be in artifacts / <raw_rootpath> / <raw_annotations>

CONFIG = {
        "raw_rootpath" : "intro_experiment/set_c/raw_samples", # Location of data we're cleaning up
        "raw_annotations" : "combined_raw_annotations.csv",
        "train_frac" : 0.6,
        "val_frac" : 0.2,
        "test_frac" : 0.2,
        "scrubbed_rootpath" : "intro_experiment/set_c/final_data" # Location of trim logs and final annotations list
        } # trimlogs always called -> samples_removed_from_raw.csv
#            final annotations always called "train_val_test_split_annotations.csv"

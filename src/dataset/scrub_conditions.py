# This file is default conditions for scrubbing data, meant to work with the default 
# Generation workflow
# All rootfile paths are relative to the project's artifacts folder. 
# The annotations file name must be in artifacts / <raw_rootpath> / <raw_annotations>
DEFAULT_CONFIG = {
        "raw_rootpath" : "raw_data", # Location of data we're cleaning up
        "raw_annotations" : "combined_raw_annotations.csv",
        "train_frac" : 0.6,
        "val_frac" : 0.2,
        "test_frac" : 0.2,
        "scrubbed_rootpath" : "final_data" # Location of trim logs and final annotations list
        } # trimlogs always called -> samples_removed_from_raw.csv
#            final annotations always called "train_val_test_split_annotations.csv"

import pandas as pd 
import matplotlib.pyplot as plt
import DoodleData as dd
import torch

def print_9_tiles(dataset : dd.CircleData, column : str, lmap : dict):
    """
    This quickly samples 9 images from the data set and prints the images within a 9 tile grid
    Mostly its for sanity checking that multiple examples/labels appear reasonable.

    Parameters:
        dataset (CircleData) : The circular image database 
        label : The label attribute we want to print for each image 
        lmap (dict) : maps the label attribute to the column it was loaded into in the CircleData class.
    """
    cols, rows = 3,3
    figure = plt.figure(figsize=(cols*rows-1, cols*rows-1))
    for i in range(1, cols*rows +1):
        samples_idx = torch.randint(len(dataset),size=(1,)).item()
        img, label = dataset[samples_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f"{column} =\n {label[lmap[column]]}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    plt.savefig(f"{column}"+ '.png')


def visualize_histograms(dataset : dd.CircleData, lmap : dict, pathtosave : str):
    """
    This function creates an len(lmap) panel png file which prints the histogram
    of each trait within the dataset split

    Parameters: 
        dataset : dd.CircleData - The set we're histogramming
        lmap : dict - the list of column names and indices loaded into dat struct
        pathtosave : str - The filename and path for the histogram png file generated 
    """
    ntiles = len(lmap)

    nrcols = 2
    nrows = 0

    if (ntiles %2 != 0): 
        # Then we'll have a blank image
        nrows = (ntiles + 1) // 2
    else: 
        nrows = ntiles//2

    records = []
    for i in range(len(dataset)):
        dict_entries = {}
        for ele in lmap:
            _, img_label = dataset[i]
            dict_entries[ele] = img_label[lmap[ele]].item()

        records.append(dict_entries)

    analysis_df = pd.DataFrame(records)

    analysis_df.hist(
            layout=(nrows, nrcols), 
            figsize=(12, nrows*4),
            bins=400,
            grid=False,
            edgecolor="black"
            )
    plt.tight_layout()
    plt.savefig(pathtosave, dpi=300)
    plt.close()

    

def split_database(data : dd.CircleData, train : float, val : float, test : float):
    """
    This function splits the master procedurally generated data into training procedural and test data depending on the fractions passed by the user. Grand data base is shuffled then drawn in without replacement

    Parameters: 
        dataset : The circle data set being divided 
        train : The % of data (decimal form) to be in the training block
        val : The % of data (decimal form) to be in the validation block
        test : The % of data (decimal form) to be in the testing data
        Note: We will normalize train/val/test params

    returns : 
        training, validation, test splits

    """
    n_total = len(data)
    norm_weight = train + val + test
    test = test / norm_weight
    train = train / norm_weight
    val = val / norm_weight

    # Shuffle og data
    indices = torch.randperm(n_total)

    n_train = int(train*n_total)
    n_val = int(val*n_total)
    n_test = int(test*n_total)

    train_split = data[indices[:n_train]]
    test_split = data[indices[n_train:(n_train+n_test)]]
    val_split = data[indices[(n_train+n_test):]]

    return train_split, val_split, test_split



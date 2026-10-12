# CircleData API

`CircleData` houses a PyTorch Dataset implementation for loading annotated .npy vector files (black and white image drawings), along with corresponding labels (e.g. ncircles) from a CSV annotatation file. It supports filtering data by dataset split, selecting specific target labels, and applying optional transformations to the feature vector (the .npy vector file). The CircleData class converts all feature and label data into PyTorch tensors for training and analysis. 

## Overview

The primary workflow is analagous to other such torch data classes. 
Data can be retrieved via indexing or by creating an iterable DataLoader from the class. 

## Usage, Initialization, Iteration

```
from DataSet.circle_data import CircleData

# Create a 1 target dataset 
train_data = CircleData(
        csv_file="csv_filepath.csv",
        target_columns=["ncircles"],
        split_tag="train"
        )

# Index DataSet or Loader
feature_0, ncircles_label_0 = train_data[0]

# Create a data loader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_data, batch_size=size, shuffle=True, num_workers=nworkers)

# Multi target dataset accessed via dict keys
# Supported keys are listed below
keys = [
    "image_size",
    "ncircles",
    "nbig",
    "nsmall",
    "area_of_circles",
    "area_of_negative",
    "max_circle_units",
    "fill_fraction",
    ]
key_dict = {key : i for i, key in enumerate(keys)}

# Create a multi target dataset 
train_data = CircleData(
        csv_file="csv_filepath.csv",
        target_columns=keys,
        split_tag="train"
        )
# Iterate as usual 
feature_0, labels_0 = train_data[0]
ncircles_0 = train_data[key_dict["ncircles"]]
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `csv_file` (req) | `str` | None. <Required> | Path to the CSV annotation file containing sample filenames, labels, paths to .npy files, and optional dataset split information |
| `root_dir`(opt) | `str` | "" | Optional directory prefix for appending to .npy file locations within database |
| `transform` (opt) | Callable or None | None | Optional transformation applied to each feature tensor after loading |
| `target_columns` (opt) | `list[str]` | `["ncircles"]` | List of CSV column names to use as target labels. Defaults to ["ncircles"] if None or an empty list is provided |
| `split_tag` (opt) | `str` or None | None | Optional dataset split identifier, when specified only rows whos data_split column matches this value are retained|

**Returns:** An initialized `CircleData` dataset object.

Initialization loads the annotation CSV into a pandas DataFrame. If split_tag is specified, the dataset is filtered to include only matching rows, and the index is reset. If filtering produces an empty dataset, a warning is printed.

The target_columns attribute defaults to ["ncircles"] when no target columns are provided.

Note: If split_tag is specified, the CSV must contain a data_split column. Each sample must also reference a valid .npy file through the access_filename column.

## Attributes
| Attribute |	Type |	Description |
|---|---|---|
| annotations	| pandas.DataFrame	| Annotation table loaded from the CSV file, optionally filtered by split_tag.|
|root_dir	| `str` |	Directory prefix used to locate .npy files.|
target_columns	| `list[str]` |	CSV columns selected as target labels. Defaults to ["ncircles"].|
transform	|Callable or None|	Optional transformation applied to input tensors.|
split_tag	| `str` or None	| Dataset split identifier used to filter the annotation table.|

## Methods
- __init__(csv_file, root_dir="", target_columns=None, transform=None, split_tag=None)

    Initializes the dataset by loading the annotation CSV, optionally filtering it by dataset split, and configuring the target columns and input transformation.

- \_\_len\_\_()

    Obtains the number of samples in the dataset after any split filtering has been applied.

    Returns: The number of available samples (int).

- \_\_getitem\_\_(idx)

    Retrieves and prepares a single dataset sample.

    The method performs the following operations:

    - Retrieves the annotation row at the specified index.
    - Constructs the path to the corresponding .npy file by joining root_dir with the row's access_filename value.
    - Loads the NumPy array using numpy.load.
    - Converts the array into a torch.float32 tensor.
    - Applies transform to the input tensor if a transformation is provided.
    - Extracts the values from the specified target_columns.
    - Converts the target values into a torch.float32 tensor.
    
    Returns: the input tensor and target tensor as a tuple.

Parameters:
Parameter	Type	Description
idx	int	Index of the sample to retrieve.

Returns: tuple[torch.Tensor, torch.Tensor] — A tuple (x, y), where:

    x is the input tensor loaded from the .npy file, optionally transformed.
    y is a one-dimensional tensor containing the selected target labels.

Note: The input tensor's shape depends on the contents of the .npy file. The target tensor has one element per selected target column.

# See Also

- [CircleDoodle](circle_doodle.md) - CircleForge's class for creating drawings and dataset entries
- torch.utils.data.Dataset — PyTorch's base class for map-style - datasets.
- torch.utils.data.DataLoader — Batches and loads samples from a dataset.
- numpy.load — Loads arrays stored in .npy files.
- pandas.read_csv — Reads CSV files into pandas DataFrames.
import pytest
import utils.io as io
import pandas as pd
from dataset.circle_data import CircleData
from test_utils.toy_database import ToyDatabase

KEYS = [
        "image_size",
        "ncircles",
        "nbig",
        "nsmall",
        "area_of_circles",
        "area_of_negative",
        "max_circle_units",
        "fill_fraction",
        ]


@pytest.fixture(scope="module") # create a fixture for all tests in file
def toy_database():
    """
    Generate the toy database once for all tests and remove afterword
    """
    db = ToyDatabase("./")
    db.generate()

    yield db # Tests run to here

    db.remove() # Post test cleanup 

@pytest.fixture(scope="module")
def circle_data_obj(toy_database):
    """
    Use generated database to create database object
    """
    return CircleData(csv_file="toy_annotations.csv", root_dir="./", target_columns=KEYS)

def test_init(circle_data_obj):
    """
    Checks expected data object structure
    """
    assert circle_data_obj is not None
    assert circle_data_obj.split_tag is None
    assert circle_data_obj.transform is None
    assert circle_data_obj.root_dir == "./"
    for i, ele in enumerate(KEYS):
        assert ele == circle_data_obj.target_columns[i]


def test_len(circle_data_obj):
    """
    Checks correct length of toy
    """
    assert len(circle_data_obj) > 0
    assert len(circle_data_obj) == len(pd.read_csv("./toy_annotations.csv"))
    assert len(circle_data_obj) == 240

def test_getitem(circle_data_obj):
    key_dict = { key : i for i, key in enumerate(KEYS)}
    img, labels = circle_data_obj[27]
    hard_coded_27 = { # keys must match KEYS (python 27 is 29 in excel)
            "image_size" : 50,
            "ncircles" : 21,
            "nbig" : 4,
            "nsmall" : 17,
            "area_of_circles" : 948,
            "area_of_negative" : 1552,
            "max_circle_units" : 5,
            "fill_fraction" : 1,
            }
    for ele in KEYS:
        assert labels[key_dict[ele]] == hard_coded_27[ele]




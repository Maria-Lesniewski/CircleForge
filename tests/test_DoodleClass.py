import pytest
import numpy as np
import sys
import os
import random
# Dynamically add the parent directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def fixture_path(request):
    """Returns the absolute path to the fixtures directory relative to the test file."""
    test_dir = os.path.dirname(request.fspath)
    return os.path.join(test_dir, "fixtures")


# Import CircleDoodle from the library file
from Procedural_Generator.generatorlib import CircleDoodle

@pytest.fixture()
def set_seed(autouse=True):
    random.seed(67)
    np.random.seed(67)

@pytest.fixture()
def default_drawing():
    return CircleDoodle()

@pytest.fixture()
def special_drawing():
    return CircleDoodle(400, 0.125, 3, "special.npy")


def test_default_init(default_drawing : CircleDoodle):
    assert default_drawing.size == 300
    assert np.sum(default_drawing.image) == 0
    assert default_drawing.image_filename == "default.npy"
    assert default_drawing.access_filename == "default.npy"
    assert np.sum(default_drawing.regionfull) == 0
    assert np.sum(default_drawing.negativespace) == 0
    assert default_drawing.ncircles == 0
    assert default_drawing.nbig == 0
    assert default_drawing.nsmall == 0
    assert default_drawing.area_of_circles == 0
    assert default_drawing.fill_fraction == 1/20
    assert default_drawing.category == "Procedural"
    assert default_drawing.max_circle_units == 200
    assert default_drawing.circle_offset == 2


def test_special_init(special_drawing : CircleDoodle):
    assert special_drawing.size == 400
    assert np.sum(special_drawing.image) == 0
    assert special_drawing.image_filename == "special.npy"
    assert special_drawing.access_filename == "special.npy"
    assert np.sum(special_drawing.regionfull) == 0
    assert np.sum(special_drawing.negativespace) == 0
    assert special_drawing.ncircles == 0
    assert special_drawing.nbig == 0
    assert special_drawing.nsmall == 0
    assert special_drawing.area_of_circles == 0
    assert special_drawing.fill_fraction == 0.125
    assert special_drawing.category == "Procedural"
    assert special_drawing.max_circle_units == 3
    assert special_drawing.circle_offset == 2


def test_is_equivalent(default_drawing):
    default_drawing.fill_space()
    assert default_drawing.is_equivalent(default_drawing) == True


def test_load_circle_obj_from_data():
    read_drawing = CircleDoodle()
    read_drawing.load_circle_obj_from_data("./fixtures/annotations.csv", "./fixtures/ones.npy")
    print(read_drawing.size)
    assert read_drawing.size == 300
    assert np.sum(read_drawing.image) == read_drawing.size*read_drawing.size
    assert np.array_equal(np.ones((300,300)), read_drawing.image) == True
    assert read_drawing.image_filename == "fixtures/ones.npy"
    assert read_drawing.access_filename == "./fixtures/ones.npy"
    assert read_drawing.ncircles == 0
    assert read_drawing.nbig == 0
    assert read_drawing.nsmall == 0
    assert read_drawing.area_of_negative_space == 90000
    assert read_drawing.max_circle_units == 200

def test_label_datapoint_with_ncircles(default_drawing):
    default_drawing.fill_space()
    default_drawing.label_datapoint_with_ncircles()
    assert default_drawing.ncircles == default_drawing.nsmall + default_drawing.nbig

def test_fill_space():
    read_drawing = CircleDoodle()
    read_drawing.load_circle_obj_from_data("./fixtures/annotations.csv", "./fixtures/fixture.npy")
    random.seed(67)
    np.random.seed(67)
    generated_drawing = CircleDoodle(size=400, filename="fixtures/fixture.npy")
    generated_drawing.fill_space()
    
    generated_drawing.label_datapoint_with_ncircles()
    generated_drawing.label_datapoints_with_area()
    assert generated_drawing.is_equivalent(read_drawing)


def test_data_base_workflow():
    doodle_obj = CircleDoodle(size=600, max_circle = 1, fill_percent = 0.01, filename="name.npy")
    doodle_obj.fill_space()
    file_exists = os.path.exists("test_data_folder")
    if not file_exists:
        os.mkdir("test_data_folder")
    doodle_obj.add_to_data_base("test_data_folder")


    doodle_obj2 = CircleDoodle(size=600)
    doodle_obj2.load_circle_obj_from_data("test_data_folder/annotations.csv","test_data_folder/name.npy")
    assert doodle_obj2.is_equivalent(doodle_obj)
    os.remove("test_data_folder/annotations.csv")
    os.remove("test_data_folder/name.npy")
    os.rmdir("test_data_folder")



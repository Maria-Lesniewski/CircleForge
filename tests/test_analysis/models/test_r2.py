import pytest 
import torch 
import math 

from analysis.models.analysis_functions.r2 import R2
from analysis.streaming_state import StreamingState
from test_utils.spy_logger import SpyLogger


## SETUP ##
@pytest.fixture
def r2_obj():
    return R2()

@pytest.fixture
def scribe():
    return SpyLogger()


## TESTS ###

def test_init(r2_obj):
    """ Basic init test """ 
    assert r2_obj.state == {
        "SSE": 0,
        "true_sum": 0,
        "true_square": 0,
        "N": 0,
    }

def test_running_accumulates_statistics(r2_obj):
    """ Checks running adds to each component for each item in batch """

    stream = StreamingState({
        "prediction_state": torch.tensor([2.0, 4.0, 5.0]),
        "label_state": torch.tensor([1.0, 3.0, 6.0]),
    })

    r2_obj.running(stream)

    assert r2_obj.state["SSE"].item() == pytest.approx(3.0)
    assert r2_obj.state["true_sum"].item() == pytest.approx(10.0)
    assert r2_obj.state["true_square"].item() == pytest.approx(46.0)
    assert r2_obj.state["N"] == 3


def test_running_accumulates_across_batches(r2_obj):
    """ Checks accumulation occurs between batches """

    stream1 = StreamingState({
        "prediction_state": torch.tensor([2.0, 4.0]),
        "label_state": torch.tensor([1.0, 3.0]),
    })

    stream2 = StreamingState({
        "prediction_state": torch.tensor([5.0]),
        "label_state": torch.tensor([6.0]),
    })

    r2_obj.running(stream1)
    r2_obj.running(stream2)

    assert r2_obj.state["SSE"].item() == pytest.approx(3.0)
    assert r2_obj.state["true_sum"].item() == pytest.approx(10.0)
    assert r2_obj.state["true_square"].item() == pytest.approx(46.0)
    assert r2_obj.state["N"] == 3


def test_final_calculates_r2(r2_obj, scribe):
    """ Tests final r2 calc on a dummy data set """

    stream = StreamingState({
        "prediction_state": torch.tensor([2.0, 4.0, 5.0]),
        "label_state": torch.tensor([1.0, 3.0, 6.0]),
    })

    r2_obj.running(stream)
    r2_obj.final(scribe)

    # SST = 46 - (10**2 / 3) = 38/3
    # R2 = 1 - (3 / (38/3)) = 29/38
    # Variance = SST / N = 38/9

    logged = scribe.add_scalars_calls["evaluation"]
    scalars = logged["scalars"]

    assert logged["step"] == 0
    assert scalars["R2"].item() == pytest.approx(29 / 38)
    assert scalars["var"].item() == pytest.approx(38 / 9)
    assert scalars["SSE"].item() == pytest.approx(3.0)
    assert scalars["SST"].item() == pytest.approx(38 / 3)
    assert scalars["N"] == 3


### EDGE CASE TESTING #####

def test_final_perfect_predictions(r2_obj, scribe):
    """ When R2 is 1 (perfect) """


    stream = StreamingState({
        "prediction_state": torch.tensor([1.0, 2.0, 3.0]),
        "label_state": torch.tensor([1.0, 2.0, 3.0]),
    })

    r2_obj.running(stream)
    r2_obj.final(scribe)

    scalars = scribe.add_scalars_calls["evaluation"]["scalars"]

    assert scalars["SSE"].item() == pytest.approx(0.0)
    assert scalars["R2"].item() == pytest.approx(1.0)


def test_final_constant_labels_returns_nan(r2_obj, scribe):
    """ When R2 is not defined (constant sets no variance) """

    stream = StreamingState({
        "prediction_state": torch.tensor([2.0, 2.0, 2.0]),
        "label_state": torch.tensor([1.0, 1.0, 1.0]),
    })

    r2_obj.running(stream)
    r2_obj.final(scribe)

    scalars = scribe.add_scalars_calls["evaluation"]["scalars"]

    assert scalars["SST"].item() == pytest.approx(0.0)
    assert scalars["SSE"].item() == pytest.approx(3.0)
    assert math.isnan(scalars["R2"])
    assert scalars["var"].item() == pytest.approx(0.0)


def test_final_constant_labels_with_perfect_predictions(r2_obj, scribe):
    """ Both perfect and invariant """ 
    stream = StreamingState({
        "prediction_state": torch.tensor([1.0, 1.0, 1.0]),
        "label_state": torch.tensor([1.0, 1.0, 1.0]),
    })

    r2_obj.running(stream)
    r2_obj.final(scribe)

    scalars = scribe.add_scalars_calls["evaluation"]["scalars"]

    assert scalars["SST"].item() == pytest.approx(0.0)
    assert scalars["SSE"].item() == pytest.approx(0.0)
    assert math.isnan(scalars["R2"])

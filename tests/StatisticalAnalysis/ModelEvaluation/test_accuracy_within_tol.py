import pytest
import torch

from StatisticalAnalysis.ModelEvaluation.analysis_functions.accuracy_within_tol import accuracy_within_tol
from StatisticalAnalysis.streaming_state import StreamingState
from test_utils.spy_logger import SpyLogger



### SET UP #### 
@pytest.fixture
def preconfigured_accuracy_within_tol():
    instance = accuracy_within_tol()
    instance.config["tol"] = 0.2
    return instance


@pytest.fixture
def scribe():
    return SpyLogger()


@pytest.fixture
def stream():
    return StreamingState({
        "prediction_state": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "label_state": torch.tensor([1.1, 2.5, 2.9, 4.1]),
        # Errors : [-0.1, -0.5, 0.1, -0.1]
        # "Correct" : 1, 0, 1, 1
    })



#############
### TESTS ###
#############

def test_init():
    """ Checks object init """ 

    preconfigured_accuracy_within_tol = accuracy_within_tol()

    assert preconfigured_accuracy_within_tol.state == {
        "num_correct": 0,
        "num_samples": 0,
    }

    assert preconfigured_accuracy_within_tol.config == {"tol": 0}


def test_running_counts_correct_predictions(preconfigured_accuracy_within_tol, scribe, stream):
    preconfigured_accuracy_within_tol.running(stream, scribe)

    # Errors: [-0.1, -0.5, 0.1, -0.1]
    # Correct: [True, False, True, True]

    assert preconfigured_accuracy_within_tol.state["num_correct"] == 3
    assert preconfigured_accuracy_within_tol.state["num_samples"] == 4

    calls = list(scribe.add_scalars_calls.values())

    assert len(calls) == 4

    expected = [True, False, True, True]

    for i, expected_correct in enumerate(expected):
        assert calls[i]["scalars"]["correct1_incorrect0"] == expected_correct
        assert calls[i]["step"] == i


def test_running_accumulates_across_batches(preconfigured_accuracy_within_tol, scribe):
    stream1 = StreamingState({
        "prediction_state": torch.tensor([1.0, 2.0]),
        "label_state": torch.tensor([1.1, 3.0]),
    })

    stream2 = StreamingState({
        "prediction_state": torch.tensor([4.0, 5.0]),
        "label_state": torch.tensor([4.1, 5.1]),
    })

    preconfigured_accuracy_within_tol.running(stream1, scribe)
    preconfigured_accuracy_within_tol.running(stream2, scribe)

    assert preconfigured_accuracy_within_tol.state["num_correct"] == 3
    assert preconfigured_accuracy_within_tol.state["num_samples"] == 4

    calls = list(scribe.add_scalars_calls.values())

    assert len(calls) == 4
    assert [call["step"] for call in calls] == [0, 1, 2, 3]


def test_running_zero_tolerance(scribe):
    preconfigured_accuracy_within_tol = accuracy_within_tol()
    preconfigured_accuracy_within_tol.config["tol"] = 0.0

    stream = StreamingState({
        "prediction_state": torch.tensor([1.0, 2.0, 3.0]),
        "label_state": torch.tensor([1.0, 2.0, 3.0]),
    })

    preconfigured_accuracy_within_tol.running(stream, scribe)

    assert preconfigured_accuracy_within_tol.state["num_correct"] == 0
    assert preconfigured_accuracy_within_tol.state["num_samples"] == 3

    calls = list(scribe.add_scalars_calls.values())

    assert all(
        call["scalars"]["correct1_incorrect0"] == False
        for call in calls
    )


def test_running_error_equal_to_tolerance_is_incorrect(scribe):
    preconfigured_accuracy_within_tol = accuracy_within_tol()
    preconfigured_accuracy_within_tol.config["tol"] = 0.5

    stream = StreamingState({
        "prediction_state": torch.tensor([1.0, 2.0]),
        "label_state": torch.tensor([1.5, 2.4]),
    })

    preconfigured_accuracy_within_tol.running(stream, scribe)

    assert preconfigured_accuracy_within_tol.state["num_correct"] == 1
    assert preconfigured_accuracy_within_tol.state["num_samples"] == 2

    calls = list(scribe.add_scalars_calls.values())

    assert calls[0]["scalars"]["correct1_incorrect0"] == False
    assert calls[1]["scalars"]["correct1_incorrect0"] == True


def test_final_writes_accuracy(preconfigured_accuracy_within_tol, scribe):
    preconfigured_accuracy_within_tol.state["num_correct"] = 3
    preconfigured_accuracy_within_tol.state["num_samples"] = 4

    preconfigured_accuracy_within_tol.final(scribe)

    assert "accuracy.json" in scribe.write_json_calls

    result = scribe.write_json_calls["accuracy.json"]["dictionary"]

    assert result == {
        "tol": 0.2,
        "num_correct": 3,
        "num_samples": 4,
        "final_%_accuracy": 75.0,
    }


def test_final_preserves_config_and_state(preconfigured_accuracy_within_tol, scribe):
    preconfigured_accuracy_within_tol.config["tol"] = 0.1
    preconfigured_accuracy_within_tol.state["num_correct"] = 8
    preconfigured_accuracy_within_tol.state["num_samples"] = 10

    preconfigured_accuracy_within_tol.final(scribe)

    result = scribe.write_json_calls["accuracy.json"]["dictionary"]

    assert result["tol"] == 0.1
    assert result["num_correct"] == 8
    assert result["num_samples"] == 10
    assert result["final_%_accuracy"] == 80.0

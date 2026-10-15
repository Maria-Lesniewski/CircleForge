import pytest
import torch

from StatisticalAnalysis.DataSetEvaluation.diagnostics_library.correlate_labels import CorrelateLabels
from test_utils.spy_logger import SpyLogger
from StatisticalAnalysis.streaming_state import StreamingState
from itertools import combinations
### SETUP ###
@pytest.fixture
def corr_obj():
    return CorrelateLabels(keys=["label1", "label2"])

@pytest.fixture
def null_corr_obj():
    return CorrelateLabels(keys=["label"])

#############
### TESTS ###
#############

## Running func test
def test_init(corr_obj, null_corr_obj):
    assert corr_obj.keys == ["label1", "label2"]
    assert null_corr_obj.keys == ["label"]
    assert null_corr_obj.running == null_corr_obj.off_running
    assert corr_obj.running == corr_obj.on_running


def test_off_running(null_corr_obj):
    logger = SpyLogger()
    stream = StreamingState({})
    null_corr_obj.running(stream, logger)
    assert len(logger.add_scalars_calls) == 0

def test_on_running_2tags(corr_obj):

    # Dummy 3 images w. labels into a stream
    features = [
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
            ]
    label1 = torch.tensor([[10.0], [20.0], [30.0]])
    label2 = torch.tensor([[0.01], [0.02], [0.03]])
    streamer = StreamingState({"features" : features, "label1" : label1, "label2" : label2})
    
    logger = SpyLogger()
    corr_obj.running(streamer, logger)

    calls = logger.add_scalars_calls

    expected_sets = list(combinations(["label1", "label2"], 2))
    expected_names = [f"{y}_vs_{x}" for x,y in expected_sets]

    assert len(calls) == 3
    for call_tag, plot_name in zip(calls, expected_names):
        assert list(calls[call_tag]["scalars"].keys())[0] == plot_name

    for i, (call_tag, (x,y)) in enumerate(zip(calls, expected_sets)):
        assert calls[call_tag]["scalars"] == {f"{y}_vs_{x}" : label2[i]}
        assert calls[call_tag]["step"] == label1[i]

def test_on_running_3tags():
    corr_obj = CorrelateLabels(keys=["label1", "label2", "label3"])

    # Dummy 3 images w. labels into a stream
    features = [
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
            ]
    label1 = torch.tensor([[10.0], [20.0], [30.0]])
    label2 = torch.tensor([[0.01], [0.02], [0.03]])
    label3 = torch.tensor([[1], [2], [3]])
    streamer = StreamingState({"features" : features, "label1" : label1, "label2" : label2, "label3": label3})

    logger = SpyLogger()
    corr_obj.running(streamer, logger)

    calls = logger.add_scalars_calls

    expected_sets = list(combinations(["label1", "label2", "label3"], 2))
    expected_names = [f"{y}_vs_{x}" for x,y in expected_sets]

    assert len(calls) == 3*3
    call_scalar_names = {list(calls[key]["scalars"].keys())[0] for key in calls }
    assert set(expected_names) == set(call_scalar_names)

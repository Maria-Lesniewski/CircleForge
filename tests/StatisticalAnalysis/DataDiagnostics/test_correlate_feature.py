import pytest
import torch

from StatisticalAnalysis.DataSetEvaluation.diagnostics_library.correlate_feature import correlate_feature
from test_utils.spy_logger import SpyLogger
from StatisticalAnalysis.streaming_state import StreamingState

### SETUP ###
@pytest.fixture
def corr_obj():
    return correlate_feature(keys=["label"])




#############
### TESTS ###
#############

### Feature Transform Tests ### 
def test_mean(corr_obj):
    feature = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    assert corr_obj.mean(feature) == pytest.approx(2.5)

def test_l2(corr_obj):
    feature = torch.tensor([
        [3.0, 4.0],
    ])

    assert corr_obj.L2(feature) == pytest.approx(5.0)


def test_laplacian_response_is_zero_for_constant_image(corr_obj):
    feature = torch.ones((5, 5))

    result = corr_obj.LP(feature)

    assert result == pytest.approx(0.0)

## Running func test

def test_running_pairs_image_mean_with_labels():
    corr_obj = correlate_feature(keys=["label"])

    # Dummy config 
    corr_obj.config = {"mean": corr_obj._feature_pooling_funcs["mean"]}
    corr_obj.state = {"mean": []}

    # Dummy 3 images w. labels into a stream
    features = [
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[2.0, 2.0], [2.0, 2.0]]),
    ]

    label = torch.tensor([[10.0], [20.0], [30.0]])

    streamer = StreamingState({"features" : features, "label" : label})
    
    # Dummy logger
    logger = SpyLogger()

    corr_obj.running(streamer, logger)

    calls = logger.add_scalars_calls

    expected_means = [0.0, 1.0, 2.0]
    expected_labels = [10.0, 20.0, 30.0]

    assert len(calls) == 3

    for call_key, mean, label in zip(calls, expected_means, expected_labels):
        assert calls[call_key]["step"] == pytest.approx(mean)
        assert calls[call_key]["scalars"] == {"label": label}


import pytest
import torch

from analysis.models.analysis_functions.histogram_error import HistogramError
from analysis.streaming_state import StreamingState

### SET UP #### 
@pytest.fixture
def preconfigured_histogram():
    """Create a configured histogram preconfigured_histogram without interactive input."""
    
    obj = HistogramError()

    obj.config = {
        "min_bin": 0.0,
        "max_bin": 4.0,
        "n_bins": 4,
    }

    # Initialize the histogram edges deterministically.
    _, edges = torch.histogram(
        torch.tensor([0.5]),
        bins=4,
        range=(0.0, 4.0),
    )
    # 0.00, 1.0, 2.0, 3.0, 4.0
    obj.state["edges"] = edges

    return obj

@pytest.fixture
def fake_stream():
    prediction_state = torch.tensor(
            [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            ]
            )
    label_state = torch.tensor(
            [
            [1.0],
            [3.0],
            [2.0],
            [-5.0],
            ]
            )
    # Errors = [ 0, -1, 1, 9]
    stream = StreamingState({"prediction_state": prediction_state, "label_state" : label_state })
    return stream

###############
### TESTS #####
###############

def test_histogram_counts(preconfigured_histogram, fake_stream):
    """Values should be assigned to the correct bins."""
    

    preconfigured_histogram.running(fake_stream)

    expected = torch.tensor([1.0, 1.0, 0.0, 0.0])

    torch.testing.assert_close(
        preconfigured_histogram.state["hist"],
        expected,
    )

def test_histogram_accumulates_across_batchs(preconfigured_histogram, fake_stream):
    batch1 = fake_stream
    prediction_state = torch.tensor(
            [
            [2.0],
            [4.0],
            [10.0],
            [8.0],
            ]
            )
    label_state = torch.tensor(
            [
            [2.0],
            [4.0],
            [7.0],
            [8.0],
            ]
            )

    batch2 = StreamingState({"prediction_state" : prediction_state, "label_state" : label_state})

    preconfigured_histogram.running(batch1)
    preconfigured_histogram.running(batch2)

    expected = torch.tensor([4.0, 1.0, 0.0, 1.0])
    torch.testing.assert_close(
        preconfigured_histogram.state["hist"],
        expected,
    )

def test_histogram_underflow_and_overflow(preconfigured_histogram, fake_stream):
    preconfigured_histogram.running(fake_stream)
    assert preconfigured_histogram.state["underflow_count"] == 1
    assert preconfigured_histogram.state["overflow_count"] == 1

def test_histogram_extrema_tracking(preconfigured_histogram, fake_stream):
    preconfigured_histogram.running(fake_stream)
    assert preconfigured_histogram.state["observed_min"] == -1
    assert preconfigured_histogram.state["observed_max"] == 9

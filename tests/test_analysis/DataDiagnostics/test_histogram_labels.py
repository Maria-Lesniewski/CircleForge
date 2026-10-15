import pytest
import torch

from analysis.DataSetEvaluation.diagnostics_library.histogram_labels import HistogramLabels
from analysis.streaming_state import StreamingState


### SET UP #### 
@pytest.fixture
def preconfigured_histogram():
    """Create a configured histogram preconfigured_histogram without interactive input."""
    
    obj = HistogramLabels(keys=["fake_data"])

    obj.config = {
        "min_bin": 0.0,
        "max_bin": 1.0,
        "n_bins": 4,
    }

    # Initialize the histogram edges deterministically.
    _, edges = torch.histogram(
        torch.tensor([0.5]),
        bins=4,
        range=(0.0, 1.0),
    )
    # 0.00, 0.25, 0.5, 0.75, 1
    obj.state["fake_data"]["edges"] = edges

    return obj

###############
### TESTS #####
###############

def test_histogram_counts(preconfigured_histogram):
    """Values should be assigned to the correct bins."""
    
    stream = StreamingState({
        "fake_data": torch.tensor([0.1, 0.2, 0.3, 0.7, 0.9])
    })

    preconfigured_histogram.running(stream)

    expected = torch.tensor([2.0, 1.0, 1.0, 1.0])

    torch.testing.assert_close(
        preconfigured_histogram.state["fake_data"]["hist"],
        expected,
    )


def test_histogram_accumulates_across_batches(preconfigured_histogram):
    """Counts should accumulate across successive calls to running()."""
    batch1 = StreamingState({
        "fake_data": torch.tensor([0.1, 0.2])
    })

    batch2 = StreamingState({
        "fake_data": torch.tensor([0.7, 0.8])
    })

    preconfigured_histogram.running(batch1)
    preconfigured_histogram.running(batch2)

    expected = torch.tensor([2.0, 0.0, 1.0, 1.0])

    torch.testing.assert_close(
        preconfigured_histogram.state["fake_data"]["hist"],
        expected,
    )


def test_histogram_underflow(preconfigured_histogram):
    """Values below min_bin should increment underflow_count."""
    stream = StreamingState({
        "fake_data": torch.tensor([-0.5, -0.1, 0.2, 0.4])
    })

    preconfigured_histogram.running(stream)

    assert preconfigured_histogram.state["fake_data"]["underflow_count"] == 2


def test_histogram_overflow(preconfigured_histogram):
    """Values above max_bin should increment overflow_count."""
    stream = StreamingState({
        "fake_data": torch.tensor([0.2, 0.4, 1.1, 1.5])
    })

    preconfigured_histogram.running(stream)

    assert preconfigured_histogram.state["fake_data"]["overflow_count"] == 2


def test_histogram_tracks_observed_extrema(preconfigured_histogram):
    """Observed extrema should include values outside the bin range."""
    stream = StreamingState({
        "fake_data": torch.tensor([-2.0, 0.3, 0.8, 3.0])
    })

    preconfigured_histogram.running(stream)

    assert preconfigured_histogram.state["fake_data"]["observed_min"] == -2.0
    assert preconfigured_histogram.state["fake_data"]["observed_max"] == 3.0


def test_histogram_includes_range_boundaries(preconfigured_histogram):
    """The minimum and maximum edges should be included in the histogram."""
    stream = StreamingState({
        "fake_data": torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    })

    preconfigured_histogram.running(stream)

    expected = torch.tensor([1.0, 1.0, 1.0, 2.0])

    torch.testing.assert_close(
        preconfigured_histogram.state["fake_data"]["hist"],
        expected,
    )

    assert preconfigured_histogram.state["fake_data"]["underflow_count"] == 0
    assert preconfigured_histogram.state["fake_data"]["overflow_count"] == 0


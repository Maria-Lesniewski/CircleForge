import pytest
import torch
from numpy import sqrt 
from StatisticalAnalysis.ModelEvaluation.analysis_functions.rmsd import RMSD
from StatisticalAnalysis.streaming_state import StreamingState
from test_utils.spy_logger import SpyLogger


## SET UP 
@pytest.fixture
def rmsd_obj():
    return RMSD()


@pytest.fixture
def scribe():
    return SpyLogger()

@pytest.fixture
def stream():
    return StreamingState({
        "prediction_state": torch.tensor([2.0, 4.0, 5.0]),
        "label_state": torch.tensor([1.0, 3.0, 7.0]),
    })

def test_init(rmsd_obj):
    assert rmsd_obj.state == {
        "sum": 0,
        "N": 0,
    }

### TESTS 
def test_running_accumulates_statistics(rmsd_obj, stream):
    """ Basic input - running """
    rmsd_obj.running(stream)

    # Squared errors: [1, 1, 4]
    # Sum of squared errors = 6
    # Number of samples = 3

    assert rmsd_obj.state["sum"].item() == pytest.approx(6.0)
    assert rmsd_obj.state["N"] == 3

def test_final_calculates_rmsd(rmsd_obj, scribe, stream):
    """ Basic input - final """
    rmsd_obj.running(stream)
    rmsd_obj.final(scribe)

    # RMSD = sqrt(sum of squared errors / N)
    #      = sqrt(6 / 3)
    #      = sqrt(2.0)

    logged = scribe.add_scalars_calls["evaluation"]

    assert logged["step"] == 0
    assert logged["scalars"]["RMSD"].item() == pytest.approx(sqrt(2.0))


def test_running_accumulates_across_batches(rmsd_obj, stream):
    """ two calls - running """
    stream2 = StreamingState({
        "prediction_state": torch.tensor([5.0]),
        "label_state": torch.tensor([6.0]),
    })

    rmsd_obj.running(stream)
    rmsd_obj.running(stream2)

    # Total squared error = 6(basic) + 1(stream2)
    # 3 (basic) + 1(stream2) : N = 4
    assert rmsd_obj.state["sum"].item() == pytest.approx(7.0)
    assert rmsd_obj.state["N"] == 4

def test_final_accumulated_batches(rmsd_obj, scribe, stream):
    """ two calls - final """

    # RMSD = sqrt(sum of squared errors / N)
    #      = sqrt(7 / 4)

    stream2 = StreamingState({
        "prediction_state": torch.tensor([5.0]),
        "label_state": torch.tensor([6.0]),
    })

    rmsd_obj.running(stream)
    rmsd_obj.running(stream2)
    rmsd_obj.final(scribe)

    logged = scribe.add_scalars_calls["evaluation"]

    assert logged["scalars"]["RMSD"].item() == pytest.approx(sqrt(7.0/4))

## Edge Case 
def test_final_with_perfect_predictions(rmsd_obj, scribe):
    stream = StreamingState({
        "prediction_state": torch.tensor([1.0, 2.0, 3.0]),
        "label_state": torch.tensor([1.0, 2.0, 3.0]),
    })

    rmsd_obj.running(stream)
    rmsd_obj.final(scribe)

    logged = scribe.add_scalars_calls["evaluation"]

    assert logged["scalars"]["RMSD"].item() == pytest.approx(0.0)



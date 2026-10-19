import pytest
from analysis.models.analysis_functions.predicted_vs_true import PredictedVSTrue
from analysis.streaming_state import StreamingState
from test_utils.spy_logger import SpyLogger

def test_predicted_vs_true():
    # Arrange
    stream = StreamingState({
        "prediction_state": [0.8, 0.3, 0.9],
        "label_state": [1.0, 0.0, 1.0],
    })

    scribe = SpyLogger()
    analysis = PredictedVSTrue()

    # Act
    analysis.running(stream, scribe)

    # Assert
    expected = [
        {"scalars": {"predicted_vs_true": 0.8}, "step": 1.0},
        {"scalars": {"predicted_vs_true": 0.3}, "step": 0.0},
        {"scalars": {"predicted_vs_true": 0.9}, "step": 1.0},
    ]

    actual = list(scribe.add_scalars_calls.values())

    assert len(actual) == len(expected)

    for actual_call, expected_call in zip(actual, expected):
        assert actual_call["scalars"].keys() == expected_call["scalars"].keys()

        for key, value in expected_call["scalars"].items():
            assert actual_call["scalars"][key] == pytest.approx(value)

        assert actual_call["step"] == expected_call["step"]

    assert scribe.add_scalars_duplicates == 2

import pytest
from analysis.ModelEvaluation.analysis_functions.error_vs_true import ErrorVSTrue
from analysis.streaming_state import StreamingState
from test_utils.spy_logger import SpyLogger

def test_error_vs_true():
    # Arrange
    stream = StreamingState({
        "prediction_state": [0.8, 0.3, 0.9],
        "label_state": [1.0, 0.0, 1.0],
    })

    scribe = SpyLogger()
    analysis = ErrorVSTrue()

    # Act
    analysis.running(stream, scribe)

    # Assert
    expected = [
        {"scalars": {"error_vs_true": -0.2}, "step": 1.0},
        {"scalars": {"error_vs_true": 0.3}, "step": 0.0},
        {"scalars": {"error_vs_true": -0.1}, "step": 1.0},
    ]

    actual = list(scribe.add_scalars_calls.values())

    assert len(actual) == len(expected)

    for actual_call, expected_call in zip(actual, expected):
        assert actual_call["scalars"].keys() == expected_call["scalars"].keys()

        for key, value in expected_call["scalars"].items():
            assert actual_call["scalars"][key] == pytest.approx(value)

        assert actual_call["step"] == expected_call["step"]

    assert scribe.add_scalars_duplicates == 2

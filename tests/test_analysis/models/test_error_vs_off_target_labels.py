import pytest
from analysis.models.analysis_functions.error_vs_off_target_labels import ErrorVSOffTargetLabels
from analysis.streaming_state import StreamingState
from test_utils.spy_logger import SpyLogger

def test_error_vs_off_target_labels():
    # Arrange
    stream = StreamingState({
        "prediction_state": [0.8, 0.3, 0.9],
        "label_state": [1.0, 0.0, 1.0],
        "class_a": [10, 20, 30],
        "class_b": [40, 50, 60],
    })

    scribe = SpyLogger()
    analysis = ErrorVSOffTargetLabels(keys=["class_a", "class_b"])

    # Act
    analysis.running(stream, scribe)

    # Assert
    expected = [
        {"scalars": {"error_vs_class_a": -0.2}, "step": 10},
        {"scalars": {"error_vs_class_a": 0.3}, "step": 20},
        {"scalars": {"error_vs_class_a": -0.1}, "step": 30},
        {"scalars": {"error_vs_class_b": -0.2}, "step": 40},
        {"scalars": {"error_vs_class_b": 0.3}, "step": 50},
        {"scalars": {"error_vs_class_b": -0.1}, "step": 60},
    ]

    actual = list(scribe.add_scalars_calls.values())

    assert len(actual) == len(expected)

    for actual_call, expected_call in zip(actual, expected):
        assert actual_call["scalars"].keys() == expected_call["scalars"].keys()

        for key, value in expected_call["scalars"].items():
            assert actual_call["scalars"][key] == pytest.approx(value)

        assert actual_call["step"] == expected_call["step"]

    assert scribe.add_scalars_duplicates == 5

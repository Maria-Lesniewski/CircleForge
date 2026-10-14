import pytest
from StatisticalAnalysis.streaming_state import StreamingState


@pytest.fixture(scope="module")
def streaming_state():
    """
    Creates a dummy streaming state obj for testing 
    """
    test_keys = {
            "int" : 120,
            "float" : 500.4,
            "list" : ["cat", "dog", "cynthia"],
            }

    return StreamingState(test_keys)

def test_init(streaming_state):
    """
    test that attributes are initialized correctly
    """
    assert streaming_state.int == 120
    assert streaming_state.float == 500.4
    assert streaming_state.list[0] == "cat"
    assert streaming_state.list[1] == "dog"
    assert streaming_state.list[2] == "cynthia"

def test_get(streaming_state):
    assert streaming_state.get("int") == 120
    assert streaming_state.get("float") == 500.4
    assert streaming_state.get("list")[0] == "cat"
    assert streaming_state.get("list")[1] == "dog"
    assert streaming_state.get("list")[2] == "cynthia"

def test_get_missing(streaming_state):
    try:
        streaming_state.get("bigfoot")
        assert False
    except AttributeError:
        assert True

def test_assign(streaming_state):
    # Verify its not there
    try:
        streaming_state.get("huckaby")
        assert False
    except AttributeError:
         assert True

    streaming_state.assign("huckaby", "Fitzgerald")
    assert streaming_state.huckaby == "Fitzgerald"
    assert streaming_state.get("huckaby") == "Fitzgerald"


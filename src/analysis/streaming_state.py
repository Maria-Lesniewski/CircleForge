# @file streaming_state.py
# @author Maria Lesniewski
# @desc This is a class for streaming important objects from datasets during data analysis

from typing import Any

class StreamingState:
    """
    This exists to make function implementations 
    seperate from any other class object that might use 
    them, while also providing a standardized-house for streaming 
    data / model predictions as necessary

    Attributes are totally custom to purpose via dict, 
    but syntax is less annoying than dict
    
    get exists so that we can use it either as dict or as attributes in object
    """
    def __init__(self, attributes : dict[str, Any]):
        self.__dict__.update(attributes)


    def get(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"{name} is not an attribute of type(self).__name__")

    def assign(self, name : str, value : Any):
        setattr(self, name, value)

# @author Maria Lesniewski
# @file analysis_registry.py
# @desc contains the analysis register class for convenient modular dev

from typing import Dict, Any

class AnalysisRegistry:
    """
    Holds a register of functions for different kinds of analysis
    along with functions for adding to and exposing the register

    Used as a class and not as an object 
    Allows totally modular function dev. 
    Functions implemented with registry decorator can be called from here 
    by linking the name of the function to its behavior via registries

    Class Attributes:
        _function_registry - basic always avaialable functions
        _special_function_registry - functions only avaialble under special conditons
    """

    _function_registry : Dict[str,Any] = {
            # Analysis registry to be filled via decorators calling 
            # the implementation below
            }
    _special_function_registry : Dict[str,Any] = {
            # Same as above, but specifically filled by
            # "special" keyword
            }

    @classmethod
    def register_analysis(cls,
            analysis_name : str, 
            special_code : str = None,
            ):
        def decorator(analysis_function):
            """
            decorator factory for adding to dictionary of implemented analysis routines
            """
            if (special_code != None):
                cls._special_function_registry[analysis_name] = analysis_function
            else:
                cls._function_registry[analysis_name] = analysis_function
            return analysis_function

        return decorator

    # Methods to expose class attributes without allowing accidental mods from user
    @classmethod
    def available(cls) -> Dict[str, Any]:
        return cls._function_registry.copy()

    @classmethod
    def special_available(cls) -> Dict[str, Any]:
        return cls._special_function_registry.copy()

    @classmethod
    def total_available(cls) -> Dict[str, Any]:
        c = cls._function_registry | cls._special_function_registry
        return c.copy()


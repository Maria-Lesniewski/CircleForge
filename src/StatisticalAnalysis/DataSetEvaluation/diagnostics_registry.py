# @author Maria Lesniewski
# @file data_diagnostics_registry.py
# @desc contains the analysis register class for convenient modular dev (like ModelEval, but for data correlations that exist between labels) 

from typing import Any

class DiagnosticsRegistry:
    """
    Holds a register of functions for different kinds of analysis
    along with functions for adding to and exposing the register

    Used as a class and not as an object 
    Allows totally modular function dev. 
    Functions implemented with registry decorator can be called from here 
    by linking the name of the function to its behavior via registries

    Class Attributes:
        _function_registry - stores list of analyses
    """

    _function_registry : dict[str,Any] = {
            # Analysis registry to be filled via decorators calling 
            # the implementation below
            }

    @classmethod
    def register_analysis(cls,
            analysis_name : str, 
            ):
        def decorator(analysis_function):
            """
            decorator factory for adding to dictionary of implemented analysis routines
            """
            cls._function_registry[analysis_name] = analysis_function
            
            return analysis_function

        return decorator

    # Methods to expose class attributes without allowing accidental mods from user
    @classmethod
    def available(cls) -> dict[str, Any]:
        return cls._function_registry.copy()

    @classmethod
    def create(cls, analysis_name : str, config : dict[str: Any]):
        """
        Factory method for diagnostics registered in the register
        Will pass args in config
        """
        supported_classes = cls.available()
        if analysis_name not in supported_classes:
                raise ValueError(f"Unknown analysis type: '{analysis_name}'")
        analysis = supported_classes[analysis_name](**config)
        return analysis


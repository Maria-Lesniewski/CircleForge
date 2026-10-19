# Analysis Registry API

## Overview

`AnalysisRegistry` provides a centralized registry for constructing analysis classes by name. 

It allows analysis code to be developed modularly and instatiated without depending directly on individual analysis implementations. 

Analysis classes are registered using the 
`@AnalysisRegistry.register_analysis()` decorated and instatiated using `AnalysisRegistry.create()`. 

The registry supports two categories of analysis:

- **Standard analysis**: Model evaluation routines that are available by default, regardless of the number of targets in the data class

- **Special Analysis**:
Model evaluation routines only enabled under special conditions, e.g. multi-objective analysis 

The class is designed to be used directly without creating an `AnalysisRegistry` instance. 


## AnalysisRegistry Methods

-  AnalysisRegistry.register_analysis(analysis_name : str, special_code : str = None)

    <details> <summary>Registers an analysis class or function in the registry.</summary>

    **Returns**

    A decorator that registers the decorated class or function and returns it unchanged.

    **Behavior**
        If special_code is provided, the analysis is registered in _special_function_registry.
        If special_code is None, the analysis is registered in _function_registry.

    **Example**

    ```
    @AnalysisRegistry.register_analysis("MyAnalysis")
    class MyAnalysis:
        def __init__(self, threshold: float = 0.5):
            self.threshold = threshold
    ```

    To register a special analysis:
    
    ```
    @AnalysisRegistry.register_analysis(
        "SpecialAnalysis",
        special_code="SPECIAL"
    )
    class SpecialAnalysis:
        def __init__(self):
            pass
    ```

    Once registered, the analysis can be instantiated through AnalysisRegistry.create().

    **Notes**
        Registration occurs when the module containing the decorated class or function is imported.
        Registering an existing name replaces the previous entry under that name in the corresponding registry.
        special_code is only used to determine which registry receives the analysis. Its actual value is not otherwise interpreted.

    </details>
    
- AnalysisRegistry.available()

    <details> <summary>Returns the standard analysis registry.</summary>

    **Parameters**

    None.

    **Returns**

    dict[str, Any]: A shallow copy of _function_registry, mapping registered analysis names to their corresponding classes or functions.

    **Example**

    ```
    analyses = AnalysisRegistry.available()
    
    for name, analysis in analyses.items():
        print(name, analysis)

    ```

    **Notes**
        Returns a copy of the registry dictionary to prevent callers from directly modifying the original registry through the returned dictionary.
        The registered classes and functions themselves are not copied.
        Special analyses are excluded.

    </details>

- AnalysisRegistry.special_available()

    <details> <summary>Returns the special analysis registry.</summary>

    **Parameters**

    None.

    **Returns**

    dict[str, Any]: A shallow copy of _special_function_registry, mapping special analysis names to their corresponding classes or functions.

    **Example**

    ```
    special_analyses = AnalysisRegistry.special_available()

    for name, analysis in special_analyses.items():
        print(name, analysis)
    ```

    **Notes**
        Returns a copy of the registry dictionary.
        Standard analyses are excluded.
        An analysis is classified as special when registered with a non-None special_code.

    </details>

- AnalysisRegistry.total_available()

    <details> <summary>Returns a combined registry containing both standard and special analyses.</summary>

    **Parameters**

    None.

    **Returns**

    dict[str, Any]: A new dictionary combining _function_registry and _special_function_registry.

    **Example**

    ```
    all_analyses = AnalysisRegistry.total_available()
    
    for name, analysis in all_analyses.items():
        print(name, analysis)
    ```

    **Notes**
        Combines the registries using dictionary union (|).
        If both registries contain the same key, the special registry entry takes precedence.
        Returns a new dictionary, so modifying it does not directly modify either original registry.

    </details>

- AnalysisRegistry.create(analysis_name : str, config : dict[str, Any] = None)

    <details> <summary>Creates an instance of a registered analysis using the provided configuration.</summary>

    **Returns**

    An instance of the requested analysis class, or the result of calling the registered function if a function was registered instead of a class.

    **Raises**
        ValueError: If analysis_name is not registered in either registry.
        TypeError: If the supplied configuration contains arguments incompatible with the constructor or registered callable.

    **Behavior**
        Replaces a missing or otherwise falsy config with an empty dictionary.
        Retrieves the combined registry using total_available().
        Checks whether analysis_name exists in the registry.
        Uses inspect.signature() to inspect the registered object's __init__ method.
        Passes config as keyword arguments if the constructor has parameters beyond self.
        Otherwise, instantiates the class without arguments.

    **Example**

    ```
    analysis = AnalysisRegistry.create(
        analysis_name="MyAnalysis",
        config={
            "threshold": 0.75
        }
    )
    ```

    For a class with no constructor arguments:

    ```
    analysis = AnalysisRegistry.create(
        analysis_name="SpecialAnalysis"
    )
    ```

    **Notes**
        Configuration keys must match the constructor's accepted parameter names.
        The implementation inspects analysis_class.__init__, so registered objects are expected to expose an appropriate __init__ method.
        The current implementation does not filter configuration arguments based on the constructor signature; it passes the entire configuration dictionary when constructor parameters beyond self exist.
        Constructor parameters with default values still count toward the signature length.

    </details>

## Registry Attributes

|Attribute| Type| 	Description|
|---|---|---|
|_function_registry|	dict[str, Any]|	Stores standard analysis classes or functions.|
|_special_function_registry|	dict[str, Any]|	Stores special analysis classes or functions.|

These are class-level dictionaries shared across all uses of AnalysisRegistry.

They are intended to be populated through register_analysis() rather than modified directly.

## Currently Registered Analyses

The registries are populated when other modules define analysis implementations using the @AnalysisRegistry.register_analysis() decorator.
Depending on the arguments passed, the analysis are added to a different register accessible via the `*available()` methods above.

### Registry Structure

|Registry|	Registration condition|	Access method|
|---|---|---|
|Standard|	special_code=None|	AnalysisRegistry.available()
|Special|	special_code is not None|	AnalysisRegistry.special_available()|
|Combined|	Includes both standard and special entries|	AnalysisRegistry.total_available()

### Analysis in Registries
|AnalysisKey| Class | Registry | Description| 
|---|---|---|---|
|"accuracy_within_tol"| `AnalysisWithinTol` |Standard| Counts how many model predictions are "correct" within a given tolerance and reports a % accuracy|
| "error_vs_true" | `ErrorVSTrue` | Standard | prints a matched (x,y) plot of the modeling error against its target objective prediction|
| "histogram_error" | `HistogramError` | Standard | Creates a binned histogram of the prediction errors according to the user-specified histogram settings|
| "predicted_vs_true" | `PredictedVSTrue` | Standard | prints a matched (x,y) plot of the model prediction against the true target label |
| "r2" | `R2` | Standard | Prints the coefficient of determination for a given model | 
| "rmsd" | RMSD | Standard | Calculates the root mean square deviation between model predictions and ground-truth labels |
|"error_vs_off_target_labels" | `ErrorVSOffTargerLabels` | Special | prints a matched (x,y) plot of the modeling error against all specified label targets|


**Design Notes**

- AnalysisRegistry is designed to be used as a class rather than instantiated as an object.

- Analysis implementations are registered through decorators.
- Standard and special analyses are stored in separate dictionaries.
- The create() method provides a centralized interface for instantiating registered analyses.

- Constructor arguments are supplied through a configuration dictionary and unpacked as keyword arguments.

- The inspect module is used to determine whether the registered class has constructor parameters beyond self.

- Registry access methods return shallow copies to prevent accidental modification of the original dictionaries.

- Duplicate names within a registry overwrite existing entries.


# See Also:
- [Analysis Library](analysis_lib.md) - Contains the complete list of classes, usage examples and API 

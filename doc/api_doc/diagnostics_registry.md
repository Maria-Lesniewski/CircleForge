# DiagnosticsRegistry API

## Overview

`DiagnosticsRegistry` provides a centralized registry for constructing diagnostic classes by name. 

It allows analysis code to be developed modularly and instatiated without depending directly on individual analysis implementations. 

Analysis classes are registered using the 
`@DiagnosticsRegistry.register_analysis()` decorated and instantiated using `DiagnosticsRegistry.create()`. 

*All analyses go into one global register*

The class is designed to be used directly without creating an `DiagnosticsRegistry` instance. 

## DiagnosticsRegistry Methods

-  DiagnosticsRegistry.register_analysis(analysis_name : str)

    <details> <summary>Registers an analysis class or function in the registry.</summary>

    **Returns**

    A decorator that registers the decorated class or function and returns it unchanged.

    **Behavior**
    
    Adds dictionary entry with {str : Class} to the registry  

    **Example**

    ```
    @DiagnosticsRegistry.register_analysis("MyAnalysis")
    class MyAnalysis:
        def __init__(self, threshold: float = 0.5):
            self.threshold = threshold
    ```

    Once registered, the analysis can be instantiated through DiagnosticsRegistry.create().

    **Notes**
        Registration occurs when the module containing the decorated class or function is imported.
        Registering an existing name replaces the previous entry under that name in the corresponding registry.
    </details>
    
- DiagnosticsRegistry.available()

    <details> <summary>Returns the standard analysis registry.</summary>

    **Parameters**

    None.

    **Returns**

    dict[str, Any]: A shallow copy of _function_registry, mapping registered analysis names to their corresponding classes or functions.

    **Example**

    ```
    analyses = DiagnosticsRegistry.available()
    
    for name, analysis in analyses.items():
        print(name, analysis)

    ```

    **Notes**
        Returns a copy of the registry dictionary to prevent callers from directly modifying the original registry through the returned dictionary.
        The registered classes and functions themselves are not copied.
    </details>

- DiagnosticsRegistry.create(analysis_name : str, config : dict[str, Any] = None)

    <details> <summary>Creates an instance of a registered analysis using the provided configuration.</summary>

    **Returns**

    An instance of the requested analysis class, or the result of calling the registered function if a function was registered instead of a class.

    **Raises**

    ValueError: If analysis_name is not registered in either registry.
    
    **Behavior**

    Retrieves registry using available().
    Checks whether analysis_name exists in the registry.
    Otherwise, instantiates the class with config arguments.

    **Example**

    ```
    analysis = DiagnosticsRegistry.create(
        analysis_name="avg_and_dev",
        config={}
    )
    ```

    **Notes**
    - Configuration keys must match the constructor's accepted parameter names.

    - The current implementation does not filter configuration arguments based on the constructor signature; it passes the entire configuration dictionary when constructor parameters beyond self exist.

    </details>

## Registry Attributes

|Attribute| Type| 	Description|
|---|---|---|
|_function_registry|	dict[str, Any]|	Stores standard analysis classes or functions.|

This is a class-level dictionary shared across all uses of DiagnosticsRegistry.

It is intended to be populated through register_analysis() rather than modified directly.

## Currently Registered Analyses

|DiagnosticsKey |	Class |	Description |
|---|---|---|
|avg_and_dev| AvgAndDev	| Prints the arithmatic mean of the dataset, the population standard deviation, the sample estimate of the standard deviation, and the standard error of the mean reported from each respectively.|
|correlate_feature| CorrelateFeature | Prints a matched (x,y) plot of data labels against feature manipulations. Currently supported feature manipulations include image L2 norms, mean, and convolutions with a 3x3 laplacian edge filter. |
|correlate_labels| CorrelateLabels | Prints a matched (x,y) plot of data labels against other data labels for multi-label data sets. |
|historgram_labels| HistogramLabels | Creates a histogram of each label in the dataset based on a global edge configuration set by the user. | 

**Design Notes**

- DiagnosticsRegistry is designed to be used as a class rather than instantiated as an object.

- Analysis implementations are registered through decorators.

- The create() method provides a centralized interface for instantiating registered analyses.

- Constructor arguments are supplied through a configuration dictionary and unpacked as keyword arguments.

- Registry access methods return shallow copies to prevent accidental modification of the original dictionaries.

- Duplicate names within a registry overwrite existing entries.


# See Also:
- [Diagnostics Library](diagnostics_lib.md) - Contains the complete list of classes, usage examples and API 

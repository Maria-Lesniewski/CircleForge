# API for Diagnostics Classes in `analysis.models.diagnostics_functions`
**Overview**

Classes in the analysis module follow one basic structure in three variations. 

- Structure: 
  - Each class contains a  `running` method that does some batchwise data analysis, this batchwise data analysis may be aggregrated into a final statistic by a `final` function 
  - Generally, `running` + `final` = overall mathematical operation. So calling `final` before `running` causes numerical problems

- Variations:
  - BasicAnalysis - One that can be done totally batchwise. Needs no data attributes for tracking
  - StatefulAnalysis - One that needs to aggregate statistics into some `state` before being finalized. Running statistics are accessed via `state` variables and aggregated during the `final` method
  - ConfigurableAnalysis - One that needs meta-information before batchwise information can be logged, such that it can be aggregated later. These functions usually have an in-built `obtain_config` function for interfacing with users on the command line, but can be configured directly through object attribute assignment.

Each analysis class employs the `Logger` class to store results. 

- ## AvgAndDev
    <details>
    <summary> Calculates the mean, standard deviation, and standard error of selected streamed values </summary>

    ### Description

    The `AvgAndDev` class calculates descriptive statistics for selected values in the streaming state. It extends `StatefulAnalysis` and is registered with `DiagnosticsRegistry` under the name `"avg_and_dev"`.

    The class accepts a list of keys identifying the values to analyze. During execution, it accumulates the number of samples, the sum of the values, and the sum of their squares for each key. Once processing is complete, the `final()` method calculates the arithmetic mean, population standard deviation, sample standard deviation, and corresponding standard error estimates.

    The population standard deviation describes the spread of the observed data, treating it as the entire population. The sample standard deviation applies Bessel's correction and estimates the population standard deviation from a sample.

    The statistics are calculated using the following formulas:

    **Arithmetic mean:**

    $$
    \mu = \frac{1}{N}\sum_{i=1}^{N}x_i
    $$

    **Population standard deviation:**

    $$
    \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2}
    $$

    **Sample standard deviation:**

    $$
    s = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(x_i-\mu)^2}
    $$

    **Standard error estimates:**

    $$
    SE_{\text{pop}} = \frac{\sigma}{\sqrt{N}}
    $$

    $$
    SE_{\text{sample}} = \frac{s}{\sqrt{N}}
    $$

    The implementation calculates the sums of squared deviations using the equivalent expression:

    $$
    \sum_{i=1}^{N}(x_i-\mu)^2
    =
    \sum_{i=1}^{N}x_i^2-N\mu^2
    $$

    The results for each key are written to a separate JSON file named `{key}_dist_stats.json`.

    ### Attributes

    |Attribute|type|desc|
    |---|---|---|
    |`state`|`dict`|Stores the accumulated statistics for each selected key. Each key maps to a dictionary containing the sample count, sum of values, and sum of squared values.|
    |`state[key]["sum_x"]`|`int` or `float`|Accumulates the sum of values associated with the key. Initially set to `0`.|
    |`state[key]["N"]`|`int`|Tracks the total number of values processed for the key. Initially set to `0`.|
    |`state[key]["sum_x2"]`|`int` or `float`|Accumulates the sum of squared values associated with the key. Initially set to `0`.|

    ### Methods

    `__init__(keys: list[str])`
    - **Description:** Initializes the `AvgAndDev` instance by calling the parent class constructor and creating an entry in `state` for each specified key. Each entry tracks the number of values, their sum, and their sum of squares.
    - **Parameters:**
        - `keys` (`list[str]`): Names of the values in the streaming state for which descriptive statistics should be calculated.
    - **Returns:** None.
    - **Example:**
    ```python
    analysis = AvgAndDev(keys=["temperature", "pressure"])

    # Initial state:
    # {
    #     "temperature": {
    #         "sum_x": 0,
    #         "N": 0,
    #         "sum_x2": 0
    #     },
    #     "pressure": {
    #         "sum_x": 0,
    #         "N": 0,
    #         "sum_x2": 0
    #     }
    # }
    ```

    `running(streamer: StreamingState, scribe: Logger, inputs: list[str] = None)`
    - **Description:** Updates the accumulated statistics for each configured key using the values in the current streaming batch.

      For each key, the method:
        1. Increments the sample count by the number of values returned by `streamer.get(key)`.
        2. Adds the sum of the values to `state[key]["sum_x"]`.
        3. Adds the sum of squared values to `state[key]["sum_x2"]`.

      The accumulated statistics allow the final descriptive statistics to be calculated across multiple batches.

    - **Parameters:**
        - `streamer` (`StreamingState`): A streaming state containing the values to analyze.
        - `scribe` (`Logger`): A logger parameter accepted by the method but not used in its implementation.
        - `inputs` (`list[str]`, optional): An optional list of inputs accepted by the method but not used in its implementation.
    - **Returns:** None. The method updates the internal state as a side effect.
    - **Example:**
    ```python
    analysis = AvgAndDev(keys=["temperature"])

    # Process a batch of values.
    analysis.running(streamer, scribe)

    # Inspect accumulated statistics.
    print(analysis.state["temperature"]["N"])
    print(analysis.state["temperature"]["sum_x"])
    print(analysis.state["temperature"]["sum_x2"])
    ```

    `final(scribe: Logger, inputs: list[str] = None)`
    - **Description:** Calculates the final descriptive statistics for each configured key and writes the results to a separate JSON file.

      For each key, the method calculates:
        - `arithmatic_mean`: Arithmetic mean of the observed values.
        - `population_stddev`: Population standard deviation.
        - `sample_stddev`: Sample standard deviation, provided that at least two values were processed.
        - `pop_SE_est`: Standard error estimate using the population standard deviation.
        - `sample_SE_est`: Standard error estimate using the sample standard deviation, provided that at least two values were processed.

      If fewer than two values have been processed, `sample_stddev` and `sample_SE_est` are set to `"Not Available"`.

      The resulting dictionary is written to `{key}_dist_stats.json` using `scribe.write_json()`.

    - **Parameters:**
        - `scribe` (`Logger`): A logger used to write the statistics to JSON files.
        - `inputs` (`list[str]`, optional): An optional list of inputs accepted by the method but not used in its implementation.
    - **Returns:** None. The method writes the calculated statistics to JSON files as a side effect.
    - **Example:**
    ```python
    analysis = AvgAndDev(keys=["temperature"])

    # Assume running() has processed the values [10, 20, 30].
    analysis.state["temperature"] = {
        "sum_x": 60,
        "N": 3,
        "sum_x2": 1400
    }

    analysis.final(scribe)

    # Writes temperature_dist_stats.json containing statistics
    # for the processed temperature values.
    ```

    ### Registration

    The class is registered with `DiagnosticsRegistry` under the name `"avg_and_dev"`.

    ```python
    @DiagnosticsRegistry.register_analysis("avg_and_dev")
    class AvgAndDev(StatefulAnalysis):
        ...
    ```

    ### Notes

    - **Selected keys:** Only keys provided during initialization are analyzed.
    - **Streaming:** Statistics accumulate across calls to `running()`.
    - **Sample size:** The sample standard deviation and its standard error estimate are unavailable when fewer than two values have been processed.
    - **Empty data:** Calling `final()` when a key has no processed values causes division by zero when calculating the mean.
    - **Numerical precision:** The variance calculation uses the difference between the sum of squares and the squared mean term. Floating-point rounding can produce small inaccuracies.
    - **Output:** Each key produces a separate `{key}_dist_stats.json` file.
    - **Implementation detail:** The output field is named `arithmatic_mean`; the spelling is preserved here to match the implementation.

    </details>

- ## CorrelateFeature
    <details>
    <summary> Analyzes relationships between selected labels and scalar transformations of input features </summary>

    ### Description

    The `CorrelateFeature` class examines relationships between selected values in the streaming state and scalar representations of input features. It extends `ConfigurableAnalysis` and is registered with `DiagnosticsRegistry` under the name `"correlate_feature"`.

    The class accepts a list of keys identifying the values to compare against the input features. Because features may contain spatial data, such as images, the class provides transformations that reduce each feature to a scalar.

    The available transformations are:

    - `mean`: Calculates the mean value of the feature.
    - `L2Norm`: Calculates the L2 norm of the feature.
    - `LaplacianResponse`: Applies a 3 × 3 Laplacian filter and calculates the mean absolute filter response.

    The `obtain_config()` method allows the user to select one or more transformations. During execution, `running()` applies each selected transformation to every feature and logs the corresponding values for each configured key.

    The logger records each selected key against the transformed feature value, allowing the relationship to be visualized as a scatter plot.

    At the end of the analysis, `final()` writes the names of the selected transformations to `feature_transform_settings.json`.

    ### Attributes

    |Attribute|type|desc|
    |---|---|---|
    |`keys`|`list[str]`|Names of the values in the streaming state to compare against transformed features.|
    |`state`|`dict`|Stores the scalar feature transformations calculated for the current batch. Each selected transformation maps to a list of scalar values.|
    |`config`|`dict`|Maps the names of selected transformations to their corresponding transformation methods. Initially empty.|
    |`choice`|`int`|An initialized attribute set to `0`. It is not used elsewhere in the current implementation.|
    |`_feature_pooling_funcs`|`dict`|Class-level mapping between transformation names and their corresponding methods: `mean`, `L2Norm`, and `LaplacianResponse`.|

    ### Methods

    `__init__(keys: list[str])`
    - **Description:** Initializes the `CorrelateFeature` instance by calling the parent class constructor. Stores the selected keys and initializes the state, configuration, and choice attributes.
    - **Parameters:**
        - `keys` (`list[str]`): Names of the values in the streaming state to compare against transformed features.
    - **Returns:** None.
    - **Example:**
    ```python
    analysis = CorrelateFeature(keys=["target_a", "target_b"])

    # Initial configuration and state.
    print(analysis.keys)
    print(analysis.config)  # {}
    print(analysis.state)   # {}
    ```

    `L2(feature: torch.Tensor) -> float`
    - **Description:** Calculates the L2 norm of the supplied feature tensor using `torch.linalg.norm()` and returns the result as a Python float.
    - **Parameters:**
        - `feature` (`torch.Tensor`): A feature tensor representing one input feature.
    - **Returns:** The L2 norm of the feature as a Python float.
    - **Example:**
    ```python
    feature = torch.tensor([[3.0, 4.0]])

    result = analysis.L2(feature)

    # result = 5.0
    ```

    `mean(feature: torch.Tensor) -> float`
    - **Description:** Calculates the mean of all values in the supplied feature tensor and returns it as a Python float.
    - **Parameters:**
        - `feature` (`torch.Tensor`): A feature tensor containing the values to average.
    - **Returns:** The mean of the feature values as a Python float.
    - **Example:**
    ```python
    feature = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = analysis.mean(feature)

    # result = 2.5
    ```

    `LP(feature: torch.Tensor) -> float`
    - **Description:** Applies a 3 × 3 Laplacian filter to a feature tensor using `torch.nn.functional.conv2d()` and returns the mean absolute filter response.

      The filter emphasizes local intensity changes, making it useful for summarizing edge-like structures in image data.

      The implementation adds batch and channel dimensions to the input tensor and applies a single-channel convolution. No padding is applied, so the spatial dimensions of the output are smaller than those of the input.
    - **Parameters:**
        - `feature` (`torch.Tensor`): A two-dimensional feature tensor representing a single image or spatial feature map.
    - **Returns:** The mean absolute Laplacian response as a Python float.
    - **Example:**
    ```python
    feature = torch.rand(10, 10)

    result = analysis.LP(feature)

    # result is a scalar Python float representing
    # the mean absolute Laplacian response.
    ```

    `obtain_config()`
    - **Description:** Interactively prompts the user to select one or more feature transformations from the available choices.

      The selected transformation methods are stored in `config`, and an empty list is initialized in `state` for each selected transformation.

      The available choices are `mean`, `L2Norm`, and `LaplacianResponse`.
    - **Parameters:** None.
    - **Returns:** None. The method updates `config` and `state` as a side effect.
    - **Example:**
    ```python
    analysis = CorrelateFeature(keys=["target_a"])

    # Prompt the user to select feature transformations.
    analysis.obtain_config()

    # If "mean" and "L2Norm" are selected:
    print(analysis.config.keys())
    # dict_keys(["mean", "L2Norm"])

    print(analysis.state)
    # {
    #     "mean": [],
    #     "L2Norm": []
    # }
    ```

    `running(streamer: StreamingState, scribe: Logger, inputs: list[str] = None)`
    - **Description:** Applies each selected transformation to every feature in the current streaming batch and logs the resulting scalar values against each configured key.

      The method performs the following operations:
        1. Clears the stored transformation results from the previous batch.
        2. Applies each selected transformation to every feature in `streamer.features`.
        3. For each configured key, logs the corresponding value from `streamer.get(key)` against each transformed feature value.

      The logging tag is formatted as `"{key}_vs_{transform}_on_feature"`, and the scalar name is the key itself. The transformed feature value is used as the logging step.

      This creates a relationship between each feature transformation and the selected streamed value.
    - **Parameters:**
        - `streamer` (`StreamingState`): A streaming state containing input features and the selected values to compare against them.
        - `scribe` (`Logger`): A logger used to record the transformed feature values and corresponding streamed values.
        - `inputs` (`list[str]`, optional): An optional list of inputs accepted by the method but not used in its implementation.
    - **Returns:** None. The method updates the internal state and logs scalar values as side effects.
    - **Example:**
    ```python
    analysis = CorrelateFeature(keys=["target_a"])

    # Select transformations before running the analysis.
    analysis.obtain_config()

    # Process the current batch.
    analysis.running(streamer, scribe)

    # Example logging tag for the "mean" transformation:
    # target_a_vs_mean_on_feature

    # The transformed feature value is the logging step,
    # and the corresponding target_a value is the logged scalar.
    ```

    `final(scribe: Logger)`
    - **Description:** Writes the selected feature transformation names to `feature_transform_settings.json`.

      The output JSON contains a `selected_functions` field listing the keys in the configuration dictionary.
    - **Parameters:**
        - `scribe` (`Logger`): A logger used to write the transformation settings to a JSON file.
    - **Returns:** None. The method writes the configuration metadata as a side effect.
    - **Example:**
    ```python
    analysis.final(scribe)

    # Writes feature_transform_settings.json containing:
    # {
    #     "selected_functions": ["mean", "L2Norm"]
    # }
    ```

    ### Registration

    The class is registered with `DiagnosticsRegistry` under the name `"correlate_feature"`.

    ```python
    @DiagnosticsRegistry.register_analysis("correlate_feature")
    class CorrelateFeature(ConfigurableAnalysis):
        ...
    ```

    ### Notes

    - **Configuration:** Call `obtain_config()` before `running()` to select transformations and initialize the corresponding state entries.
    - **Feature input:** `LP()` expects a two-dimensional feature tensor for a single feature map. Its convolution uses no padding.
    - **Logging:** Results are logged through `Logger.add_scalars()`. This logger writes scalar data to Grace-compatible `.xvg` files and TensorBoard.
    - **Batch behavior:** Transformation results are cleared at the beginning of every `running()` call. The state stores results for the current batch rather than accumulating them across batches.
    - **Output:** The selected transformation names are saved to `feature_transform_settings.json`.
    - **Implementation detail:** Despite its name, the class does not calculate a statistical correlation coefficient. It logs paired values for visualization, allowing relationships to be inspected externally.

    </details>

- ## CorrelateLabels
    <details>
    <summary> Logs pairwise relationships between selected streamed values </summary>

    ### Description

    The `CorrelateLabels` class logs pairs of selected values from the streaming state to help visualize their relationships. It extends `BasicAnalysis` and is registered with `DiagnosticsRegistry` under the name `"correlate_labels"`.

    The class accepts a list of keys identifying the values to compare. It generates all unique pairs of keys using `itertools.combinations()` and logs the values of each pair using `scribe.add_scalars()`.

    For each pair `(x, y)`, the class logs the value associated with `y` using the scalar name `"{y}_vs_{x}"` and the corresponding value associated with `x` as the logging step. This allows the resulting data to be visualized as a scatter plot.

    If fewer than two keys are provided, the class disables the analysis by assigning `off_running()` to `self.running`. In that case, calls to `running()` do nothing.

    ### Attributes

    |Attribute|type|desc|
    |---|---|---|
    |`keys`|`list[str]`|Names of the values in the streaming state to compare pairwise.|
    |`running`|`method`|Assigned during initialization to either `on_running` or `off_running`, depending on the number of keys provided.|

    ### Methods

    `__init__(keys: list[str])`
    - **Description:** Initializes the analysis and stores the selected keys. Assigns `on_running()` as the active `running()` method when at least two keys are supplied. If fewer than two keys are supplied, assigns `off_running()` instead and prints a warning explaining that the analysis will be skipped.

      The constructor also prints the selected keys to the console.
    - **Parameters:**
        - `keys` (`list[str]`): Names of the values in the streaming state to compare pairwise.
    - **Returns:** None.
    - **Example:**
    ```python
    analysis = CorrelateLabels(keys=["temperature", "pressure", "humidity"])

    # All unique pairs will be compared:
    # temperature and pressure
    # temperature and humidity
    # pressure and humidity
    ```

    `off_running(stream: StreamingState, scribe: Logger, inputs: list[str] = None)`
    - **Description:** Does nothing. This method is assigned as the active `running()` method when fewer than two keys are provided.
    - **Parameters:**
        - `stream` (`StreamingState`): A streaming state parameter accepted by the method but not used.
        - `scribe` (`Logger`): A logger parameter accepted by the method but not used.
        - `inputs` (`list[str]`, optional): An optional list of inputs accepted by the method but not used.
    - **Returns:** None.
    - **Example:**
    ```python
    analysis = CorrelateLabels(keys=["temperature"])

    # The analysis is disabled because only one key was provided.
    analysis.running(stream, scribe)

    # No values are logged.
    ```

    `on_running(stream: StreamingState, scribe: Logger, inputs: list[str] = None)`
    - **Description:** Generates all unique pairs of configured keys and logs their corresponding values.

      For each pair `(x, y)`, the method iterates over the values returned by `stream.get(x)`. At each index `i`, it logs the value `stream.get(y)[i]` using the scalar name `"{y}_vs_{x}"` and uses `stream.get(x)[i]` as the logging step.

      The logging category is `"diagnostic"`.
    - **Parameters:**
        - `stream` (`StreamingState`): A streaming state containing the selected values.
        - `scribe` (`Logger`): A logger used to record the pairwise values.
        - `inputs` (`list[str]`, optional): An optional list of inputs accepted by the method but not used.
    - **Returns:** None. The method logs scalar values as a side effect.
    - **Example:**
    ```python
    analysis = CorrelateLabels(keys=["temperature", "pressure"])

    # Suppose:
    # stream.get("temperature") = [10, 20, 30]
    # stream.get("pressure") = [1, 2, 3]

    analysis.running(stream, scribe)

    # Logs under the "diagnostic" category:
    # Scalar name: "pressure_vs_temperature"
    # Logging steps: 10, 20, 30
    # Logged values: 1, 2, 3
    ```

    ### Registration

    The class is registered with `DiagnosticsRegistry` under the name `"correlate_labels"`.

    ```python
    @DiagnosticsRegistry.register_analysis("correlate_labels")
    class CorrelateLabels(BasicAnalysis):
        ...
    ```

    ### Notes

    - **Minimum keys:** At least two keys are needed to generate pairwise comparisons.
    - **Pair generation:** Each unique pair is generated once. The class does not generate both `(x, y)` and `(y, x)` separately.
    - **Logging:** Values are logged under the `"diagnostic"` category. The logging step is the value of the first key in each pair, rather than a sample index.
    - **Output:** The class logs scalar values but does not write a final JSON report.
    - **Implementation detail:** The class visualizes pairwise relationships; it does not calculate a numerical correlation coefficient.

    </details>

- ## HistogramLabels
    <details>
    <summary> Builds histograms of selected label values and tracks observed ranges and out-of-range counts </summary>

    ### Description

    The `HistogramLabels` class generates running histograms for selected values in the streaming state. It extends `ConfigurableAnalysis` and is registered with `DiagnosticsRegistry` under the name `"histogram_labels"`.

    The class accepts a list of keys identifying the values to analyze. A single histogram configuration is shared across all selected keys, specifying the lower boundary, upper boundary, and number of bins.

    The `obtain_config()` method prompts the user to supply the histogram boundaries and bin count. During execution, `running()` computes a histogram for each selected key and accumulates the bin counts across batches. It also tracks the observed minimum and maximum values, along with the number of values below or above the configured range.

    At the end of the analysis, `final()` writes the shared configuration and per-key metadata to JSON files and logs each histogram's bin counts.

    ### Attributes

    |Attribute|type|desc|
    |---|---|---|
    |`keys`|`list[str]`|Names of the values in the streaming state for which histograms are generated.|
    |`config`|`dict`|Stores the shared histogram configuration for all selected keys.|
    |`config["max_bin"]`|`float`|Upper boundary of the histogram range. Values greater than this boundary contribute to the overflow count.|
    |`config["min_bin"]`|`float`|Lower boundary of the histogram range. Values below this boundary contribute to the underflow count.|
    |`config["n_bins"]`|`int`|Number of bins used to divide the configured range.|
    |`state`|`dict`|Maps each selected key to its histogram data and metadata.|
    |`state[key]["hist"]`|`torch.Tensor` or `int`|Accumulated histogram counts for the selected key. Initially set to `0`.|
    |`state[key]["edges"]`|`torch.Tensor` or `int`|Histogram bin edges generated during configuration. Initially set to `0`.|
    |`state[key]["observed_min"]`|`float`|Minimum value observed for the key across processed batches. Initially set to positive infinity.|
    |`state[key]["observed_max"]`|`float`|Maximum value observed for the key across processed batches. Initially set to negative infinity.|
    |`state[key]["overflow_count"]`|`int`|Number of values greater than `max_bin`. Initially set to `0`.|
    |`state[key]["underflow_count"]`|`int`|Number of values less than `min_bin`. Initially set to `0`.|

    ### Methods

    `__init__(keys: list[str])`
    - **Description:** Initializes the `HistogramLabels` instance by calling the parent class constructor, storing the selected keys, and initializing the shared histogram configuration. Creates a separate histogram state dictionary for each key.
    - **Parameters:**
        - `keys` (`list[str]`): Names of the values in the streaming state to analyze.
    - **Returns:** None.
    - **Example:**
    ```python
    analysis = HistogramLabels(keys=["temperature", "pressure"])

    # Initial configuration:
    # {
    #     "max_bin": 0,
    #     "min_bin": 0,
    #     "n_bins": 0
    # }

    # Each key has its own histogram state.
    print(analysis.state.keys())
    # dict_keys(["temperature", "pressure"])
    ```

    `running(stream: StreamingState, scribe: Logger = None, keys: list[str] = None)`
    - **Description:** Computes and accumulates histogram counts and metadata for each configured key using the current batch.

      For each key, the method:
        1. Retrieves the corresponding values using `stream.get(key)`.
        2. Computes a histogram using the configured number of bins and range.
        3. Adds the batch histogram counts to the accumulated counts.
        4. Increments the overflow count for values greater than `max_bin`.
        5. Increments the underflow count for values less than `min_bin`.
        6. Updates the observed minimum and maximum values when new extremes are encountered.

      The method processes every configured key using the same histogram boundaries and bin count.

    - **Parameters:**
        - `stream` (`StreamingState`): A streaming state containing the values to analyze.
        - `scribe` (`Logger`, optional): A logger parameter accepted by the method but not used in its implementation.
        - `keys` (`list[str]`, optional): An optional list of keys accepted by the method but not used. The class uses `self.keys` instead.
    - **Returns:** None. The method updates the internal histogram state as a side effect.
    - **Example:**
    ```python
    analysis = HistogramLabels(keys=["temperature"])

    # Configure the histogram before processing data.
    analysis.config["min_bin"] = 0.0
    analysis.config["max_bin"] = 100.0
    analysis.config["n_bins"] = 10

    # Process a batch of values.
    analysis.running(stream)

    # Inspect the accumulated histogram and metadata.
    print(analysis.state["temperature"]["hist"])
    print(analysis.state["temperature"]["observed_min"])
    print(analysis.state["temperature"]["observed_max"])
    print(analysis.state["temperature"]["overflow_count"])
    print(analysis.state["temperature"]["underflow_count"])
    ```

    `final(scribe: Logger)`
    - **Description:** Writes the histogram configuration and metadata to JSON files and logs the histogram counts for each selected key.

      For each key, the method:
        1. Writes the shared configuration to `histogram_config.json`.
        2. Writes the observed minimum and maximum values and overflow and underflow counts to `{key}_histogram_meta.json`.
        3. Logs each histogram bin count under the category `"evaluation_{key}"`, using the corresponding bin edge as the logging step.

      The metadata JSON excludes the `hist` and `edges` entries from the internal state.

      Because `histogram_config.json` is written once for each key, the same file is overwritten repeatedly when multiple keys are configured. Its final contents are the shared configuration and are identical regardless of which key writes it.
    - **Parameters:**
        - `scribe` (`Logger`): A logger used to write JSON files and log histogram counts.
    - **Returns:** None. The method writes metadata and logs histogram values as side effects.
    - **Example:**
    ```python
    analysis.final(scribe)

    # Output files:
    # histogram_config.json
    # temperature_histogram_meta.json
    # pressure_histogram_meta.json

    # Histogram counts are logged under:
    # evaluation_temperature
    # evaluation_pressure
    ```

    `obtain_config()`
    - **Description:** Interactively obtains and validates the shared histogram configuration.

      The method repeatedly prompts the user for `max_bin`, `min_bin`, and `n_bins`. It checks that the boundary values are floats and that the bin count is an integer. Once the input types are valid, it stores the configuration and generates histogram bin edges for each selected key.

      If the input types are invalid, the method displays an error message and prompts the user to try again.
    - **Parameters:** None.
    - **Returns:** None. The method updates `config` and initializes the `edges` entry for each key.
    - **Example:**
    ```python
    analysis = HistogramLabels(keys=["temperature", "pressure"])

    # Prompt the user for histogram boundaries and bin count.
    analysis.obtain_config()

    # Inspect the validated configuration.
    print(analysis.config)

    # Inspect the bin edges for each key.
    print(analysis.state["temperature"]["edges"])
    print(analysis.state["pressure"]["edges"])
    ```

    ### Registration

    The class is registered with `DiagnosticsRegistry` under the name `"histogram_labels"`.

    ```python
    @DiagnosticsRegistry.register_analysis("histogram_labels")
    class HistogramLabels(ConfigurableAnalysis):
        ...
    ```

    ### Notes

    - **Configuration:** Call `obtain_config()` or manually set `config` before processing batches.
    - **Shared range:** All selected keys use the same histogram boundaries and number of bins.
    - **Accumulation:** Histogram counts and metadata accumulate across calls to `running()`.
    - **Overflow and underflow:** The overflow and underflow checks use strict comparisons (`>` and `<`). Values exactly equal to the configured boundaries are not counted as overflow or underflow.
    - **Output:** The configuration is written to `histogram_config.json`, and each key produces a separate `{key}_histogram_meta.json` file.
    - **Implementation detail:** The prompt text refers to `"histogram_error"` even though this class is registered as `"histogram_labels"`.
    - **Validation:** The method checks input types but does not verify that `max_bin > min_bin` or that `n_bins` is positive.
    - **Empty data:** If no values are processed for a key, its observed minimum and maximum remain positive and negative infinity, respectively. Calling `final()` before successful configuration may also fail because histogram edges and counts have not been initialized correctly.

    </details>
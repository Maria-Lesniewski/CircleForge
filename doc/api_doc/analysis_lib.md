# API for Analysis Classes in analysis.models.analysis_functions

- ## AccuracyWithinTol
    <details>
    <summary> Calculates a model's percent accuracy within a given tolerance for a specified label prediction </summary>
    
    ### Description

    `AccuracyWithinTol` is a configurable analysis class that evaluates prediction accuracy by determining whether the absolute error between predictions and their corresponding labels is strictly less than a specified tolerance (`tol`).

    The class tracks the number of correct predictions and the total number of samples processed. During execution, it logs individual prediction results and, when evaluation is complete, calculates the final accuracy percentage and saves the results to `accuracy.json`.

    A prediction is considered correct if:

    $$|\text{prediction} - \text{label}| < \text{tol}$$

    **Note:** The comparison is strictly less than (`<`), meaning predictions with an absolute error equal to the tolerance are considered incorrect.

    ### Attributes

    |Attribute|type|desc|
    |---|---|---|
    |`state`|`dict`|Stores the running evaluation statistics. Initially contains `num_correct` (the number of correct predictions) and `num_samples` (the total number of processed samples), both initialized to `0`.|
    |`config`|`dict`|Stores the analysis configuration. Contains `tol`, the tolerance threshold for determining whether a prediction is correct, initialized to `0.01`.|

    ### Methods

    `__init__()`
    - **Description:** Initializes the analysis by calling the parent class constructor and setting up the initial evaluation state and configuration.
    - **Parameters:** None.
    - **Returns:** None.
    - **Example:**
    ```python
    analysis = AccuracyWithinTol()

    # Initial state:
    # {
    #     "num_correct": 0,
    #     "num_samples": 0
    # }

    # Initial configuration:
    # {
    #     "tol": 0.01
    # }
    ```

    `running(stream: StreamingState, scribe: Logger = None)`
    - **Description:** Processes a batch of predictions and labels from the streaming state. Calculates the prediction errors and determines whether each prediction falls within the configured tolerance. Updates the running counts of correct predictions and total samples, and logs each prediction's correctness inside the `evaluation_correct1_incorrect0` data set, which indexes each samples and denotes its category with either a one for correct under the tolerance or a zero for incorrect under the tolerance.
    - **Parameters:**
        - `stream` (`StreamingState`): The streaming state containing `prediction_state` and `label_state`, which hold the predictions and corresponding ground-truth labels.
        - `scribe` (`Logger`): The logger used to record the correctness of individual predictions.
    - **Returns:** None.
    - **Example:**
        ```python
        analysis = AccuracyWithinTol()
        analysis.config["tol"] = 0.5

        # Assume stream.prediction_state and stream.label_state
        # contain the following values:
        #
        # Predictions: [1.2, 2.1, 3.8]
        # Labels:      [1.0, 2.0, 4.0]
        #
        # Errors:      [0.2, 0.1, -0.2]
        # Correct:     [True, True, True]

        analysis.running(stream, scribe)

        # Updated state:
        # {
        #     "num_correct": 3,
        #     "num_samples": 3
        # }
        ```

    `final(scribe: Logger)`
    - **Description:** Finalizes the evaluation by combining the configuration and accumulated state into a result dictionary, calculating the final accuracy percentage, and writing the results to `accuracy.json`. The output includes the configured tolerance, the total number of correct predictions, the total number of samples, and the final accuracy percentage.
    - **Parameters:**
        - `scribe` (`Logger`): The logger used to write the final evaluation results as JSON, with the relative filepath `"accuracy.json"`.
    - **Returns:** None.
    - **Example:**
    ```python
    # Assume the analysis has processed 10 samples,
    # of which 8 were correct, with a tolerance of 0.5.

    analysis.config["tol"] = 0.5
    analysis.state["num_correct"] = 8
    analysis.state["num_samples"] = 10

    analysis.final(scribe)

    # Writes accuracy.json containing:
    # {
    #     "tol": 0.5,
    #     "num_correct": 8,
    #     "num_samples": 10,
    #     "final_%_accuracy": 80.0
    # }
    ```

    `obtain_config()`
    - **Description:** Interactively requests the tolerance value from the user. Repeatedly prompts for input until a valid floating-point value is provided, then stores the value in `self.config["tol"]`. If the provided value is invalid, displays an error message and asks the user to try again.
    - **Parameters:** None.
    - **Returns:** None.
    - **Example:**
    ```python
    analysis = AccuracyWithinTol()
    analysis.obtain_config()

    # Example user input:
    # tol = 0.5
    #
    # If the input is a valid float:
    # analysis.config["tol"] = 0.5
    #
    # If the input is invalid:
    # An error message is displayed, and the user
    # is prompted to try again.
    ```

    ### Registration

    The class is registered with `AnalysisRegistry` under the name `"accuracy_within_tol"`, allowing it to be identified and retrieved through the analysis registry.

    ```python
    @AnalysisRegistry.register_analysis("accuracy_within_tol")
    class AccuracyWithinTol(ConfigurableAnalysis):
        ...
    ```

    ### Notes

    - **Tolerance requirement:** The tolerance should be configured before processing samples. The default value is `0.01` which may not be suitable for all inference targets.
    - **Accuracy calculation:** The final accuracy is calculated as `(num_correct / num_samples) * 100`.
    - **Empty dataset:** Calling `final()` when `num_samples` is `0` raises a `ZeroDivisionError`.
    - **Logging:** The `running()` method logs each sample's correctness using `scribe.add_scalars()`, with the sample index based on the number of previously processed samples.
    - **Output:** The final evaluation results are written to `accuracy.json`.

    </details>

- ## ErrorVSOffTargetLabels
    <details>
    <summary> Prints a matched XY plot of signed model error vs. non-model targetted data labels </summary>
    
    ### Description

    The `ErrorVSOffTargetLabels` class analyzes the difference between model predictions and ground-truth labels, comparing the prediction error against other labels in the dataset. It extends `BasicAnalysis` and is registered with `AnalysisRegistry` under the name `"error_vs_off_target_labels"` with the category `"special"`.

    During execution, the class iterates over the specified keys and prediction states, calculates the prediction error by subtracting the ground-truth label from the prediction, and logs the error against the corresponding values retrieved from the streaming state using `scribe.add_scalars()`.

    ### Attributes

    |Attribute|type|desc|
    |---|---|---|
    |`keys`|`list[str]`| A list of keys identifying the target labels or values in the streaming state against which the prediction error is compared.|

    ### Methods

    `__init__(keys: list[str])`
    - **Description:** Initializes the `ErrorVSOffTargetLabels` instance and stores the specified keys. Calls the parent class constructor to initialize the base analysis functionality.
    - **Parameters:**
        - `keys`: A list of strings identifying the labels or values to compare against the model prediction error.
    - **Returns:** None.
    - **Example:**
    ```python
    analysis = ErrorVSOffTargetLabels(keys=["label_a", "label_b"])
    ```

    `running(stream: StreamingState, scribe: Logger)`
    - **Description:** Computes and logs the difference between model predictions and ground-truth labels for each specified key. For every key, the method iterates through the prediction states, calculates the prediction error as `prediction_state[i] - label_state[i]`, and logs the result under the `"evaluation"` category. Each error is associated with the corresponding value retrieved using `stream.get(key)[i]`.

      The logged scalar name follows the format `"error_vs_{key}"`, allowing prediction errors to be analyzed in relation to different labels or values in the dataset.
    - **Parameters:**
        - `stream`: A `StreamingState` object containing the model predictions, ground-truth labels, and other relevant data.
        - `scribe`: A `Logger` instance used to record the computed prediction errors.
    - **Returns:** None. The method logs scalar values as a side effect.
    - **Example:**
    ```python
    analysis = ErrorVSOffTargetLabels(keys=["label_a", "label_b"])

    # Execute the analysis using the streaming state and logger.
    analysis.running(stream, scribe)

    # Logged scalar names include:
    # "error_vs_label_a"
    # "error_vs_label_b"
    ```

    </details>

- ## ErrorVSTrue
    <details>
    <summary> Prints a matched XY plot of signed model error vs. true targetted values </summary>
    
    ### Description

    The `ErrorVSTrue` class analyzes the difference between model predictions and ground-truth labels. It extends `BasicAnalysis` and is registered with `AnalysisRegistry` under the name `"error_vs_true"`.

    During execution, the class iterates through the prediction states, calculates the prediction error by subtracting the ground-truth label from the model prediction, and logs the error against the corresponding ground-truth value using `scribe.add_scalars()`.

    This analysis helps evaluate how prediction errors vary with the true values in the dataset.

    ### Attributes

    |Attribute|type|desc|
    |---|---|---|
    |Inherited attributes|`BasicAnalysis`|Attributes provided by the parent class `BasicAnalysis`. (No attributes)|

    ### Methods

    `running(stream: StreamingState, scribe: Logger)`
    - **Description:** Computes and logs the difference between model predictions and ground-truth labels for each prediction state. The prediction error is calculated as `stream.prediction_state[i] - stream.label_state[i]`. Each error is logged under the `"evaluation"` category with the scalar name `"error_vs_true"` and the corresponding ground-truth value as the logging step or x-axis value.
    - **Parameters:**
        - `stream`: A `StreamingState` object containing the model predictions and ground-truth labels.
        - `scribe`: A `Logger` instance used to record the computed prediction errors.
    - **Returns:** None. The method logs scalar values as a side effect.
    - **Example:**
    ```python
    # Create an instance of the analysis.
    analysis = ErrorVSTrue()

    # Execute the analysis using the streaming state and logger.
    analysis.running(stream, scribe)

    # For each prediction, the logged scalar is:
    # error_vs_true = prediction_state[i] - label_state[i]
    # The corresponding label_state[i] is used as the logging step.
    ```

    </details>

- ## HistogramError
    <details>
    <summary> Creates histogram of signed prediction errors </summary>
    
    ### Description

    The `HistogramError` class generates a running histogram of signed prediction errors, calculated as the difference between model predictions and ground-truth labels. It extends `ConfigurableAnalysis` and is registered with `AnalysisRegistry` under the name `"histogram_error"`.

    The histogram is computed using user-defined boundaries and a specified number of bins. The class accumulates histogram counts across batches while tracking the observed minimum and maximum errors, the number of errors exceeding the upper bound (overflow), and the number of errors falling below the lower bound (underflow).

    The analysis requires configuration parameters provided through the `obtain_config()` method. At the end of the analysis, the `final()` method saves the configuration and histogram metadata as JSON files and logs the histogram counts for visualization.

    ### Attributes

    |Attribute|type|desc|
    |---|---|---|
    |`state`|`dict`|Stores the internal histogram state, including histogram counts, bin edges, observed minimum and maximum errors, and overflow and underflow counts.|
    |`state["hist"]`|`torch.Tensor` or `int`|Stores the accumulated histogram counts across batches. Initially set to `0` and updated during each call to `running()`.|
    |`state["edges"]`|`torch.Tensor` or `int`|Stores the histogram bin edges generated during configuration. Initially set to `0`.|
    |`state["observed_min"]`|`float`|Tracks the minimum prediction error observed across all processed batches. Initially set to positive infinity.|
    |`state["observed_max"]`|`float`|Tracks the maximum prediction error observed across all processed batches. Initially set to negative infinity.|
    |`state["overflow_count"]`|`int`|Counts the number of prediction errors greater than `max_bin` that fall outside the histogram's upper boundary.|
    |`state["underflow_count"]`|`int`|Counts the number of prediction errors less than `min_bin` that fall outside the histogram's lower boundary.|
    |`config`|`dict`|Stores the user-defined histogram configuration.|
    |`config["max_bin"]`|`float`|The upper boundary of the histogram range. Errors greater than this value contribute to the overflow count.|
    |`config["min_bin"]`|`float`|The lower boundary of the histogram range. Errors less than this value contribute to the underflow count.|
    |`config["n_bins"]`|`int`|The number of bins used to divide the histogram range between `min_bin` and `max_bin`.|

    ### Methods

    `__init__()`
    - **Description:** Initializes the `HistogramError` instance by calling the parent class constructor. Creates the internal `state` dictionary for tracking histogram counts, bin edges, observed error bounds, and overflow and underflow counts. Also initializes the `config` dictionary with default values for the histogram boundaries and number of bins.
    - **Parameters:** None.
    - **Returns:** None.
    - **Example:**
    ```python
    analysis = HistogramError()

    # Initial configuration
    print(analysis.config)
    # {
    #     "max_bin": 0,
    #     "min_bin": 0,
    #     "n_bins": 0
    # }

    # Initial histogram state
    print(analysis.state["observed_min"])  # inf
    print(analysis.state["observed_max"])  # -inf
    ```

    `running(stream: StreamingState, scribe: Logger = None)`
    - **Description:** Computes and accumulates a histogram of signed prediction errors for the current batch.

      The method performs the following operations:
        1. Calculates prediction errors as `stream.prediction_state - stream.label_state`.
        2. Computes a histogram using the configured number of bins and range.
        3. Adds the batch histogram counts to the accumulated histogram.
        4. Counts errors above `max_bin` as overflows and errors below `min_bin` as underflows.
        5. Updates the observed minimum and maximum prediction errors when new extremes are encountered.

      The histogram counts and metadata are accumulated across calls, allowing the analysis to process multiple batches.
    - **Parameters:**
        - `stream`: A `StreamingState` object containing the model predictions and ground-truth labels.
        - `scribe`: An optional `Logger` instance. This parameter is not used within this method.
    - **Returns:** None. The method updates the internal histogram state as a side effect.
    - **Example:**
    ```python
    analysis = HistogramError()

    # Configure the histogram before processing data.
    analysis.config["min_bin"] = -1.0
    analysis.config["max_bin"] = 1.0
    analysis.config["n_bins"] = 20

    # Process a batch of predictions and labels.
    analysis.running(stream)

    # Inspect the accumulated histogram and metadata.
    print(analysis.state["hist"])
    print(analysis.state["observed_min"])
    print(analysis.state["observed_max"])
    print(analysis.state["overflow_count"])
    print(analysis.state["underflow_count"])
    ```

    `final(scribe: Logger)`
    - **Description:** Finalizes the analysis by saving the histogram configuration and metadata to JSON files and logging the histogram counts.

      The method performs the following operations:
        1. Saves the histogram configuration to `histogram_config.json`.
        2. Saves the histogram metadata, including observed minimum and maximum errors and overflow and underflow counts, to `histogram_meta.json`.
        3. Logs each histogram bin count under the `"evaluation"` category, using the corresponding bin edge as the logging step.

      The metadata JSON excludes the `hist` and `edges` entries from the internal state.
    - **Parameters:**
        - `scribe`: A `Logger` instance used to write JSON files and log histogram values.
    - **Returns:** None. The method saves results and logs histogram values as side effects.
    - **Example:**
    ```python
    analysis.final(scribe)

    # Output files:
    # histogram_config.json
    # histogram_meta.json

    # Histogram counts are logged under the "evaluation" category.
    # Each count is associated with its corresponding bin edge.
    ```

    `obtain_config()`
    - **Description:** Interactively obtains and validates the histogram configuration from the user.

      The method repeatedly prompts the user to provide values for `max_bin`, `min_bin`, and `n_bins`. It validates that the boundary values are floats and the number of bins is an integer. Once the inputs are valid, it stores them in `self.config` and generates the histogram bin edges using `torch.histogram()` with randomly generated sample data.

      If the input types are invalid, the method displays an error message and prompts the user to try again.
    - **Parameters:** None.
    - **Returns:** None. The method updates the `config` dictionary and initializes `state["edges"]`.
    - **Example:**
    ```python
    analysis = HistogramError()

    # Prompt the user to enter:
    # max_bin: upper histogram boundary
    # min_bin: lower histogram boundary
    # n_bins: number of histogram bins
    analysis.obtain_config()

    # Inspect the validated configuration.
    print(analysis.config)

    # Inspect the generated histogram bin edges.
    print(analysis.state["edges"])
    ```

    </details>

- ## PredictedVSTrue
    <details>
    <summary> Prints a matched X,Y plot of the predicted model value vs. the actual data label </summary>

    ### Description

    The `PredictedVSTrue` class logs model predictions against their corresponding ground-truth labels. It extends `BasicAnalysis` and is registered with `AnalysisRegistry` under the name `"predicted_vs_true"`.

    During execution, the class iterates through the prediction states and logs each model prediction under the `"evaluation"` category. The corresponding ground-truth label is used as the logging step or x-axis value.

    This analysis helps visualize the relationship between predicted values and actual labels, making it easier to evaluate model performance and identify patterns in prediction behavior.

    ### Attributes

    |Attribute|type|desc|
    |---|---|---|
    |Inherited attributes|`BasicAnalysis`|Attributes provided by the parent class `BasicAnalysis`. (No Attributes)|

    ### Methods

    `running(stream: StreamingState, recorder: Logger)`
    - **Description:** Iterates through the model predictions and logs each prediction against its corresponding ground-truth label. Each prediction is recorded under a data file the name `"evaluation_predicted_vs_true"`. The corresponding value from `stream.label_state` is used as the logging step (x-axis).
    - **Parameters:**
        - `stream`: A `StreamingState` object containing the model predictions and ground-truth labels.
        - `recorder`: A `Logger` instance used to record the predicted values.
    - **Returns:** None. The method logs scalar values as a side effect.
    - **Example:**
        ```python
        # Create an instance of the analysis.
        analysis = PredictedVSTrue()

        # Execute the analysis using the streaming state and logger.
        analysis.running(stream, recorder)

        # Each prediction is logged under "predicted_vs_true",
        # using the corresponding ground-truth label as the logging step.
        ```

    </details>

- ## RMSD
    <details>
    <summary> Calculates the root mean square deviation between model predictions and ground-truth labels</summary>
    
    ### Description

    The `RMSD` class calculates the root mean square deviation (RMSD) between model predictions and ground-truth labels. It extends `StatefulAnalysis` and is registered with `AnalysisRegistry` under the name `"rmsd"`.

    RMSD measures the magnitude of prediction errors by calculating the square root of the mean of the squared differences between predicted and actual values. Squaring the errors ensures that positive and negative errors do not cancel each other out, while larger errors contribute more heavily to the final metric.

    The class maintains a running sum of squared errors and the total number of predictions processed across batches. Once processing is complete, the `final()` method calculates the RMSD and logs the result under the `"evaluation"` category.

    The RMSD is calculated using the following formula:

    $$
    \text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2}
    $$

    Where:
    - $N$: Total number of predictions.
    - $\hat{y}_i$: Model prediction for sample $i$.
    - $y_i$: Ground-truth value for sample $i$.

    ### Attributes

    |Attribute|type|desc|
    |---|---|---|
    |`state`|`dict`|Stores the cumulative statistics required to calculate RMSD across multiple batches.|
    |`state["sum"]`|`int` or `torch.Tensor`|Accumulates the sum of squared differences between predictions and ground-truth labels. Initially set to `0`.|
    |`state["N"]`|`int`|Tracks the total number of predictions processed across all batches. Initially set to `0`.|

    ### Methods

    `__init__()`
    - **Description:** Initializes the `RMSD` instance by calling the parent class constructor and creating the internal `state` dictionary. The dictionary stores the cumulative sum of squared prediction errors and the total number of predictions processed.
    - **Parameters:** None.
    - **Returns:** None.
    - **Example:**
    ```python
    analysis = RMSD()

    # Initial state
    print(analysis.state)
    # {
    #     "sum": 0,
    #     "N": 0
    # }
    ```

    `running(streamer: StreamingState, scribe: Logger = None)`
    - **Description:** Updates the internal state using the predictions and ground-truth labels from the current batch.

      The method performs the following operations:
        1. Increments `state["N"]` by the number of predictions in the batch.
        2. Calculates the squared difference between each prediction and its corresponding ground-truth label.
        3. Adds the sum of squared differences to `state["sum"]`.

      By accumulating these values across batches, the method enables RMSD to be calculated over the entire dataset rather than separately for each batch.
    - **Parameters:**
        - `streamer`: A `StreamingState` object containing the model predictions and ground-truth labels.
        - `scribe`: An optional `Logger` instance. This parameter is not used within this method.
    - **Returns:** None. The method updates the internal state as a side effect.
    - **Example:**
        ```python
        analysis = RMSD()

        # Process a batch of predictions and labels.
        analysis.running(streamer)

        # Inspect the accumulated statistics.
        print(analysis.state["sum"])  # Sum of squared errors
        print(analysis.state["N"])    # Number of predictions processed
        ```

    `final(recorder: Logger)`
    - **Description:** Calculates the final RMSD using the accumulated sum of squared errors and the total number of predictions. The result is logged under the `"evaluation"` category with the scalar name `"RMSD"` and a logging step of `0`.

      The calculation is performed as follows:

      $$
      \text{RMSD} = \sqrt{\frac{\text{state[``sum"]}}{\text{state[``N"]}}}
      $$

      This method assumes that at least one prediction has been processed.
    - **Parameters:**
        - `recorder`: A `Logger` instance used to record the final RMSD value.
    - **Returns:** None. The method logs the calculated RMSD as a side effect.
    - **Example:**
        ```python
        analysis = RMSD()

        # Process one or more batches.
        analysis.running(streamer)

        # Calculate and log the final RMSD.
        analysis.final(recorder)

        # The RMSD value is logged under:
        # Category: "evaluation"
        # Scalar name: "RMSD"
        # Logging step: 0
        ```

    </details>


- ## R2
    <details>
    <summary> Calculates coefficient of determination to determine how well model predictions explain variance in actual labels</summary>

    ### Description

    The `R2` class calculates the coefficient of determination $R^2$ to evaluate how well the model predictions fit the ground-truth values. It extends `StatefulAnalysis` and is registered with `AnalysisRegistry` under the name `"r2"`.

    The coefficient of determination measures the proportion of variance in the ground-truth values that is explained by the model. It is calculated using the residual sum of squares (SSE) and the total sum of squares (SST).

    The class accumulates the necessary statistics across batches, including the sum of squared prediction errors, the sum of ground-truth values, the sum of squared ground-truth values, and the total number of samples. Once processing is complete, the `final()` method calculates $R^2$, the variance of the ground-truth values, and related statistics. These values are logged under the `"evaluation"` category.

    The coefficient of determination is calculated using the following formula:

    $$
    R^2 = 1 - \frac{SSE}{SST}
    $$

    Where:

    - $SSE$: Sum of squared errors between model predictions and ground-truth values.
    - $SST$: Total sum of squares, representing the variation of ground-truth values around their mean.

    The total sum of squares is calculated as:

    $$
    SST = \sum_{i=1}^{N} y_i^2 - \frac{(\sum_{i=1}^{N} y_i)^2}{N}
    $$

    The variance of the ground-truth values is calculated as:

    $$
    \text{var} = \frac{SST}{N}
    $$

    **Note:** Although $R^2$ is often between 0 and 1 for a well-fitting model on a training dataset, it can be negative when the model performs worse than predicting the mean. If all ground-truth values are identical, the implementation returns `NaN` because $SST = 0$.

    ### Attributes

    |Attribute|type|desc|
    |---|---|---|
    |`state`|`dict`|Stores the cumulative statistics required to calculate the coefficient of determination across multiple batches.|
    |`state["SSE"]`|`int` or `torch.Tensor`|Accumulates the sum of squared errors between model predictions and ground-truth values. Initially set to `0`.|
    |`state["true_sum"]`|`int` or `torch.Tensor`|Accumulates the sum of all ground-truth values processed. Used to calculate the total sum of squares. Initially set to `0`.|
    |`state["true_square"]`|`int` or `torch.Tensor`|Accumulates the sum of squared ground-truth values. Used to calculate the total sum of squares. Initially set to `0`.|
    |`state["N"]`|`int`|Tracks the total number of samples processed across all batches. Initially set to `0`.|

    ### Methods

    `__init__()`
    - **Description:** Initializes the `R2` instance by calling the parent class constructor and creating the internal `state` dictionary. The dictionary stores the cumulative sum of squared errors, the sum of ground-truth values, the sum of squared ground-truth values, and the total number of samples.
    - **Parameters:** None.
    - **Returns:** None.
    - **Example:**
        ```python
        analysis = R2()

        # Initial state
        print(analysis.state)
        # {
        #     "SSE": 0,
        #     "true_sum": 0,
        #     "true_square": 0,
        #     "N": 0
        # }
        ```

    `running(stream: StreamingState, scribe: Logger = None)`
    - **Description:** Updates the internal statistics using the predictions and ground-truth values from the current batch.

      The method performs the following operations:
        1. Increments `state["N"]` by the number of predictions in the batch.
        2. Calculates the squared differences between predictions and ground-truth values and adds their sum to `state["SSE"]`.
        3. Adds the sum of ground-truth values to `state["true_sum"]`.
        4. Adds the sum of squared ground-truth values to `state["true_square"]`.

      These accumulated statistics allow the final $R^2$ value to be calculated across all processed batches rather than independently for each batch.
    - **Parameters:**
        - `stream`: A `StreamingState` object containing the model predictions and ground-truth values.
        - `scribe`: An optional `Logger` instance. This parameter is not used within this method.
    - **Returns:** None. The method updates the internal state as a side effect.
    - **Example:**
        ```python
        analysis = R2()

        # Process a batch of predictions and ground-truth values.
        analysis.running(stream)

        # Inspect the accumulated statistics.
        print(analysis.state["SSE"])         # Sum of squared errors
        print(analysis.state["true_sum"])    # Sum of ground-truth values
        print(analysis.state["true_square"]) # Sum of squared ground-truth values
        print(analysis.state["N"])           # Number of samples processed
        ```

    `final(scribe: Logger)`
    - **Description:** Calculates the final coefficient of determination and related statistics using the accumulated state.

      The method performs the following operations:
        1. Calculates the total sum of squares (`SSTot`) using the accumulated ground-truth statistics.
        2. Calculates the variance of the ground-truth values as `SSTot / N`.
        3. Calculates $R^2$ as `1 - SSE / SSTot` when `SSTot` is nonzero. If `SSTot` is zero, the method assigns `float("nan")` to $R^2$.
        4. Logs the calculated metrics under the `"evaluation"` category at logging step `0`.
        5. Prints the final $R^2$ value to the console.

      The logged metrics include:
        - `R2`: The coefficient of determination.
        - `var`: The variance of the ground-truth values, calculated using a population variance formula.
        - `SSE`: The sum of squared prediction errors.
        - `SST`: The total sum of squares.
        - `N`: The total number of samples processed.
    - **Parameters:**
        - `scribe`: A `Logger` instance used to record the final evaluation metrics.
    - **Returns:** None. The method logs the metrics and prints the final \(R^2\) value as side effects.
    - **Example:**
        
        ```python
        analysis = R2()

        # Process one or more batches.
        analysis.running(stream)

        # Calculate and log the final evaluation metrics.
        analysis.final(scribe)

        # Logged metrics under "evaluation*" file/dataset names:
        # R2  - Coefficient of determination
        # var - Variance of ground-truth values
        # SSE - Sum of squared prediction errors
        # SST - Total sum of squares
        # N   - Number of samples processed

        # Console output:
        # Final r2 <calculated_value>
        ```
    </details>

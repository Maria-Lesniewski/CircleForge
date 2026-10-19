# Analysis Tools — User Guide

This project provides two tools for examining datasets and evaluating trained models: `DiagnosticsRunner` and `AnalysisRunner`.

I anticipate reworking each in the future, so I've not documented the in depth structure of each herein. 

Briefly:

Each class pairs a methods factory ([DiagnosticsRegistry](diagnostics_registry.md), [AnalysisRegistry](analysis_registry.md)) with a configurable question from the [user interface library](io.md) to obtain a schedule of different functions that can be calculated simultaneously while iteracting through a data set. 

Each runner has a slightly different purpose: data trend analysis (DiagnosticsRunner) or Model Evaluation (AnalysisRunner).

**Which should I use?**
- Use DiagnosticsRunner to investigate your dataset.
- Use AnalysisRunner to investigate your trained model's predictions and performance.


<a id="DiagnosticsRunner"></a>

## DiagnosticsRunner

Use `DiagnosticsRunner` to analyze patterns in a dataset, such as relationships between features and labels, without running a model.

**Basic usage:**
1. Create a runner with your dataset.
2. Select the analyses you want to perform.
3. Run the analyses.
4. Close the runner when finished.

```python
from ... import DiagnosticsRunner

runner = DiagnosticsRunner(
    dataset=my_dataset,
    filewriter_root_path="artifacts"
)

runner.obtain_schedule()
runner.run_schedule()
runner.close()
```

<a id="AnalysisRunner"></a>

## AnalysisRunner

Use AnalysisRunner to evaluate a trained model against a dataset. It runs the model on the data and uses its predictions to perform the selected analyses.

Basic usage:

1.    Provide a trained model and dataset.
2.    Specify the target label the model predicts.
3.   Select the analyses you want to perform.
4.    Run the analyses.
5.    Close the runner when finished.

```python
from ... import AnalysisRunner

runner = AnalysisRunner(
    model=my_model,
    dataset=my_dataset,
    filewriter_root_path="artifacts",
    evaluate_against="target_name"
)

runner.obtain_schedule()
runner.run_schedule()
runner.close()
```

## Output and Notes

- *Analysis selection*: Both runners prompt you to select from the available analysis routines. Some routines may request additional configuration.
- *Output*: Results are logged under the project's artifacts directory, organized by analysis type and dataset split. Model analyses are also organized by model class.
- *Batch processing*: Both runners process the dataset in batches, with a default batch size of 32.
- *Dataset requirements*: The dataset must provide target_columns and split_tag attributes and yield batches containing features and labels.
- *Cleanup*: Call close() after running analyses to close the associated loggers.



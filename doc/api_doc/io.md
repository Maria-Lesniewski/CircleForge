# `io.py` API Reference

Provides reusable I/O functions for driver scripts and establishes a stable anchor for project path references.

## Overview

The `io.py` module provides utilities for:
- Interactive command-line questions and user input.
- Yes/no, single-choice, and multiple-choice prompts.
- Collecting dictionary values from user input.
- Resolving project-relative paths.
- Creating and locating artifact directories and files.
- Deleting files using absolute paths.
- Identifying the directory containing a calling file.

## Command-Line Interface Functions
- yes_no_q(extra_info: str, question: str) -> bool
    <details>
    <summary>Prompts the user with a yes/no question and requires a valid response.</summary>

    **Parameters**
    - extra_info (str): Contextual information printed before the question.
    - question (str): The yes/no question presented to the user.

    **Returns:**
    - bool: True if the user enters "Y".
    - bool: False if the user enters "N".

    **Behavior:**

    Displays the contextual information and question.
    Accepts only uppercase "Y" or "N".
    Repeats the prompt if the response is invalid.

    **Example:**
    ```
    if yes_no_q("This action cannot be undone.", "Continue?"):
        print("Continuing...")
    else:
        print("Operation cancelled.")
    ```
    </details>

- multiple_choice_q(extra_info: str, multiple_choice: list[str])
    <details>
    <summary>  Prompts the user to select a single option from a list.</summary>

    **Parameters:**
    - extra_info (str): Contextual information displayed before the options.
    - multiple_choice (list[str]): A list of available options.

    **Returns:**
    - str: The selected option from multiple_choice.

    **Behavior:**

    Displays each option alongside its zero-based integer index.
    Accepts an integer corresponding to an option.
    Repeats the prompt if the response is not an integer or is outside the valid range.

    **Example:**

    ```
    options = ["Option A", "Option B", "Option C"]

    selection = multiple_choice_q(
        "Choose an option.",
        options
    )

    print(f"You selected: {selection}")
    ``` 

    **Notes:**

    Option indices start at 0.
    The function uses NumPy to generate the valid integer indices.
    An empty options list is not supported by the current implementation.
    </details>

- compounding_multiple_choice_q(extra_info: str, multiple_choice: list[str])
    <details>
    <summary> Prompts the user to select multiple options from a list. </summary>

    The user must enter the selected option indices separated by spaces, followed by the sentinel value -1.

    **Parameters:**

    - extra_info (str): Contextual information displayed before the options.
    - multiple_choice (list[str]): A list of available options.

    **Returns:**

    - list[str]: The selected options, with duplicate selections removed.

    **Behavior:**
    Displays each option alongside its zero-based integer index.
    Accepts multiple integer indices separated by spaces.
    Requires -1 as the final input value to indicate completion.
    Repeats the prompt if the input contains non-integer values or invalid indices.
    Exits the program with status code 1 if an unexpected exception occurs in the selection-processing block.

    **Example:**

    ```python
    options = ["Option A", "Option B", "Option C"]

    selections = compounding_multiple_choice_q(
        "Select all applicable options.",
        options
    )

    print(f"Selected options: {selections}")

    # Example user input:

    # 0 2 -1

    # Example return value:

    # ["Option A", "Option C"]
    ``` 
    **Notes:**
    The sentinel -1 must appear as the final integer.
    All preceding integers must correspond to valid option indices.
    The returned list is constructed using list(set(selected)), so the original selection order is not guaranteed.
    An empty options list is not supported by the current implementation.
    </details>

- dictionary_filling_question(extra_info: str, keys: list[str]) -> dict[str, Any]
    <details>
    <summary>Prompts the user to enter a value for each specified dictionary key.</summary>

    **Parameters:**

    - extra_info (str): Contextual information displayed before the prompts.
    - keys (list[str]): The list of dictionary keys for which values must be collected.

    **Returns:**

    - dict[str, Any]: A dictionary mapping each key to the corresponding user-entered string.

    **Behavior:**
    Displays the contextual information.
    Prompts the user to enter a value for each key.
    Stores each response in a dictionary.
    Returns the completed dictionary.

    **Example:**
    ```python

    keys = ["name", "age", "occupation"]

    user_data = dictionary_filling_question(
        "Please enter your information.",
        keys
    )

    print(user_data)

    # Example user input:

    # name: Yoda 
    # age: 900
    # occupation: Jedi Master

    #Example return value:

    #{
    #    "name": "Yoda",
    #    "age": "900",
    #    "occupation": "Jedi Master"
    #}
    ``` 
    **Notes:**
    All values are returned as strings, regardless of their apparent data type.
    Type validation and conversion are the responsibility of the calling program.
    No validation is performed on empty responses.
    </details>
## Path Management Functions

- make_artifact_dir(name: str) -> Path
    <details>
    <summary>Creates a directory inside the project's artifacts directory and returns its path.</summary>

    Parameters:

        name (str): The name or relative path of the directory to create within ARTIFACTS.

    Returns:

        Path: The path to the created or existing directory.

    Behavior:

        Constructs the directory path using ARTIFACTS / name.
        Creates the directory and any missing parent directories.
        Does not raise an error if the target directory already exists.

    **Examples:**

    ```python
    output_dir = make_artifact_dir("results")

    print(output_dir)

    #Example return value:

    # usr/documents/CircleForge/artifacts/results

    experiment_dir = make_artifact_dir("experiments/run_01")
    
    # This creates the experiments/run_01 directory under artifacts,
    # including any missing parent directories.

    ``` 

    **Notes:**
    Directory creation is performed using Path.mkdir(parents=True, exist_ok=True).
    The function does not create a regular file.
    The caller is responsible for supplying an appropriate directory name or relative path.
    </details>

- artifact_path(name: str) -> Path
    <details>
        <summary>Constructs a path relative to the project's artifacts directory without creating the corresponding file or directory.</summary>

        **Parameters:**
        - name (str): The filename or relative path within the artifacts directory.

        **Returns:**
        - Path: The constructed artifact path.

        **Behavior:**
        Joins ARTIFACTS with the supplied name.
        Does not check whether the target exists.
        Does not create files or directories.

        **Examples:**

        ```python
        output_file = artifact_path("results/output.csv")

        print(output_file)

        # Example return value:

        #usr/Documents/CircleForge/artifacts/results/output.csv

        # Usage with file operations:

        import pandas as pd

        output_file = artifact_path("results/output.csv")

        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "score": [90, 85]
        })

        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        ```

        **Notes:**
        - Use make_artifact_dir() when a directory needs to be created.
        - Use artifact_path() when a path needs to be constructed.
        - The function does not guarantee that the returned path is contained within ARTIFACTS if an absolute path or path traversal components are supplied.
    </details>

- find_project_root() -> Path
    <details>
    <summary>Returns the project's root directory.</summary>

    **Parameters:**
        None.

    **Returns:**
        Path: The absolute path stored in PROJECT_ROOT.

    **Behavior:**
        Returns the module-level PROJECT_ROOT constant.
        Does not perform a new filesystem search.

    **Example:**

    ```python
    project_root = find_project_root()

    print(project_root)

    # Example return value:

    # '/usr/Documents/CircleForge/`
    ```

    **Usage:**

    ```python
    config_file = find_project_root() / "config" / "settings.json"
    ```
    </details>

- delete_file_abs_path(abs_file_path: str) -> bool

    <details>
    <summary>Deletes a file at the specified absolute path if it exists.</summary>

    **Parameters:**
    - abs_file_path (str): The absolute path of the file to delete.

    **Returns:**
    - bool: True if the path does not exist after the operation.
    - bool: False if the path still exists after the operation.

    **Behavior:**Checks whether the specified path exists.
    If it exists, attempts to delete it using Path.unlink().
    Checks whether the path exists after the deletion attempt.
    Returns the result of that final existence check.

    **Example:**

    ```python
    file_path = "/tmp/output.txt"

    success = delete_file_abs_path(file_path)

    if success:
        print("File successfully removed or did not exist.")
    else:
        print("File still exists.")
    ```
    **Notes:**
    - The function does not suppress filesystem exceptions. For example, a permission error may cause Path.unlink() to raise an exception.
    - The function does not verify that the supplied path is absolute, despite the parameter name.
    - The function is intended for files and symbolic links, not directories.
    - If the target does not exist initially, the function returns True.
    - If the path refers to a directory, the deletion attempt may raise an exception.
    </details>

- find_here() -> Path

    <details>
    
    <summary>Finds the absolute directory path of the file that directly calls find_here().</summary>

    **Parameters:**
    - None.

    **Returns:**
    - Path: The resolved absolute path to the calling file's parent directory.

    **Behavior:**
    Inspects the current Python call stack.
    Retrieves the filename associated with the immediate caller.
    Resolves the caller's parent directory to an absolute path.
    Returns that directory path.

    **Example:**

    Suppose the project contains the following files:

    project/
    ├── src/
    │   └── utilities/
    │       └── io.py
    └── scripts/
        └── run.py

    In run.py:
    ```python
    from src.utilities.io import find_here

    script_dir = find_here()

    print(script_dir)

    # Example return value:
 
    # usr/docs/CircleForge/scripts
    ``` 

    **Notes:**

    - The function returns the calling file's directory, not the path to the calling file itself.
    - It uses inspect.stack()[1].filename to identify the immediate caller.
    - The result is resolved to an absolute path.
    - Stack inspection may introduce overhead compared with simpler path-resolution approaches.

## Function Summary

|Function	| Return Type|	Description|
|---|---|---|
|yes_no_q(extra_info, question)	| bool	|Prompts the user for a yes/no response.|
|multiple_choice_q(extra_info, multiple_choice)|str |	Prompts the user to select one option.|
| compounding_multiple_choice_q(extra_info, multiple_choice)|list[str]|	Prompts the user to select multiple options.|
|dictionary_filling_question(extra_info, keys)|	dict[str, Any]	|Collects user input into a dictionary.|
|make_artifact_dir(name)|	Path	|Creates and returns an artifact directory.|
|artifact_path(name)	|Path	| Constructs an artifact path without creating it.|
| find_project_root()	| Path|	Returns the project root directory.|
| delete_file_abs_path(abs_file_path)	| bool|	Attempts to delete a file and reports whether its path remains.|
|find_here()|	Path|	Returns the directory containing the immediate calling file.|

## Usage Guidelines
- *User interaction*: Use the question functions to collect command-line input consistently across driver scripts.
- *Input validation*: Validate or convert returned values when the calling program requires specific types or constraints.
- *Artifact creation*: Use make_artifact_dir() to create output directories.
- *Artifact path construction*: Use artifact_path() to reference files within the artifacts directory without creating them.
- *Project-relative paths*: Use find_project_root() to construct paths relative to the project root.
- *Caller-relative paths*: Use find_here() when a script needs to locate resources relative to its own directory.
- *File deletion*: Use delete_file_abs_path() when a file should be removed, accounting for possible filesystem exceptions.
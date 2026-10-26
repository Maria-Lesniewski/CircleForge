# Trainer API

*I anticipate breaking Trainer apart into a smaller easier to configure interface in future releases. 
In V1, the only epoch_routing registered is the `best_checkpoint_epochs` routine
*

## Overview

`Trainer` provides a centralized interface for managing PyTorch model training and validation routines.

It bundles the model, optimizer, loss function, data loaders, and logger into a single class, allowing different training epoch routines to be selected without excessive argument passing in the main experimental scripts. 

Training routines are registered using the 
`@Trainer.register_epoch()` decorator and selected by name when intializng a `Trainer` instance. 

The training process is started using `Trainer.train()`, which executes the selected epoch routine using the provided configuration. 

## Trainer Methods

- Trainer.register_epoch(epoch_routine_name : str)

    <details> 
    <summary>Registers a training epoch routine in the class-level registry.</summary>

    **Returns**

    A decorator that registers the decorated function under epoch_routine_name and returns the function unchanged.

    **Example**

    ```
    @Trainer.register_epoch("my_epoch_routine")
    def my_epoch_routine(self, n_epochs: int):
        for epoch in range(n_epochs):
            # Training logic
            pass
    ```

    Once registered, the routine can be selected when constructing a Trainer instance.

    **Notes**

        The registry is shared across all Trainer instances.
        Registered routines are stored in Trainer._epoch_registry.
        The decorated routine receives the Trainer instance as its first argument.

    </details>

- Trainer.\_\_init\_\_(model_config : [ModelConfig](model_config.md) , optimizer : torch.optim.Optimizer, loss_fn : torch.nn.Module , train_loader : DataLoader, val_loader : DataLoader, logger : Logger, epoch_style : str)

    <details> 
    <summary>Initializes a trainer with the required training components and selects an epoch routine.</summary>

    **Returns**

    An initialized Trainer instance.

    **Raises**

        ValueError: If epoch_style does not match a registered epoch routine.

    **Notes**

        Automatically selects CUDA if available; otherwise, uses the CPU.
        Moves the model to the selected device.
        Stores the provided optimizer, loss function, data loaders, and logger.
        Resolves the selected epoch routine from Trainer._epoch_registry.

    **Example**

    ```
    trainer = Trainer(
        model_config=model_config,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        epoch_style="best_checkpoint_epochs"
    )
    ```
    </details>


- Trainer.train(epoch_config : Dict[str, Any])

    <details> <summary> Logs the trainer's setting and executes the selected epoch routine using the provided configuration.</summary>

    **Returns**

    None

    Behavior

        Writes the trainer's settings to training_settings.json through the logger.
        Executes the selected epoch routine with the arguments provided in epoch_config.

    **Notes**

        The keys in epoch_config must match the parameter names expected by the selected epoch routine.
        The Trainer instance is passed automatically to the routine as its first argument.
        Invalid or missing routine arguments will raise the corresponding Python TypeError.

    **Example**

    ```
    trainer.train(
        epoch_config={
            "n_epochs": 10,
            "validation_interval": 100
        }
    )
    ```
    </details>

- Trainer._get_epoch_routine(epoch_style : str)

    <details> <summary>Retrieves a registered epoch routine by name.</summary>

    **Returns**

    The function associated with the specified registry key.

    **Raises**

        ValueError: If the specified epoch routine is not registered.

    **Notes**

        This is an internal method called during initialization.
        The returned function is stored in self.epoch_routine and invoked by Trainer.train().

    **Example**
    ```
    routine = trainer._get_epoch_routine("best_checkpoint_epochs")
    ```
    </details>

- Trainer.best_checkpoint_epochs(n_epochs : int, validation_interval : int)

    <details> <summary>Trains the model for a specified number of epochs, periodically evaluates validation loss, and saves checkpoints when validation performance improves.</summary>

    **Returns**

    None

    **Raises**

        ValueError: If validation_interval is greater than the number of training batches in an epoch.

    **Behavior**

        Iterates through the training data for n_epochs.
        Moves each input batch and its labels to the selected device.
        Clears the optimizer's gradients.
        Performs a forward pass through the model.
        Calculates the loss using loss_fn.
        Performs backpropagation and updates the model parameters.
        Logs the training loss for each batch.
        Evaluates the model on the validation dataset at the specified intervals.
        Logs the average training and validation losses.
        Saves a model checkpoint when the average validation loss improves.

    
    *Checkpoint Selection*

    The routine initializes best_validation_loss to 10**30. A checkpoint is saved whenever the current average validation loss is lower than the best validation loss recorded during the current call.

    Checkpoints are saved through logger.save_model() and include the model, training step, validation loss, training loss, and metadata.

    **Example**

    ```
    trainer.train(
        epoch_config={
            "n_epochs": 20,
            "validation_interval": 50
        }
    )
    ```

    **Notes**

        Validation occurs after every validation_interval batches, using zero-based batch indices.
        Validation is performed at the end of each complete interval; any remaining batches at the end of an epoch do not trigger an additional validation.
        The average training loss reported for each validation interval is calculated across the batches in that interval.
        The epoch's average training loss is printed at the end of each epoch.
        The routine assumes that train_loader and val_loader contain at least one batch.
        The model is returned to training mode after each validation evaluation.

    </details>

- Trainer.evaluate_model()

    <details> <summary>Evaluates the model over the validation dataset and calculates the average validation loss.</summary>

    **Returns**

    float: Average validation loss across all batches in val_loader.

    **Behavior**

        Switches the model to evaluation mode using model.eval().
        Disables gradient calculation using torch.no_grad().
        Iterates over the validation dataset.
        Moves validation inputs and labels to the selected device.
        Performs a forward pass and calculates the loss for each batch.
        Computes the average loss across all validation batches.
        Restores the model to training mode using model.train().

    **Example**

    ```
    validation_loss = trainer.evaluate_model()
    print(f"Validation loss: {validation_loss}")
    ```

    **Notes**

        Does not perform backpropagation or optimizer updates.
        Calculates the arithmetic mean of the batch loss values, rather than weighting each batch by its number of samples.
        Assumes val_loader contains at least one batch.
        Always switches the model back to training mode after evaluation, regardless of its mode before the call.
    </details>

## Currently Registered Epoch Routines

|Registry Key| Method | Description |
|---|---|---|
|"best_checkpoint_epochs"|	best_checkpoint_epochs|	Trains for a specified number of epochs, periodically evaluates validation loss, and saves checkpoints when validation performance improves.|

**Design Notes**

- Epoch routines are registered in the class-level _epoch_registry dictionary.
- Each epoch routine must be registered using @Trainer.register_epoch("routine_name").
- The registered routine must accept the Trainer instance as its first argument.
- Epoch routines are selected by name through the epoch_style constructor parameter.
- Additional epoch routines can be implemented without changing the main training interface.
- The train() method accepts routine-specific arguments through a dictionary, which is unpacked into keyword arguments.
- The model is automatically moved to CUDA when available, or CPU otherwise.
- Training metrics and model checkpoints are recorded through the provided logger.
- The optimizer is expected to reference the parameters of the model supplied to the trainer.

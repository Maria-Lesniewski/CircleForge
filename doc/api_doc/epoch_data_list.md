
# Epoch Data List 
## Overview

Each epoch routine logs scalar functions to tensorboard and to .xvg grace files as it tracks metrics. Tensorboard stores data with the same tag under one scalars window, while grace seperates each scalar into its own (x,y) style file.
 
Below we list each grace file name (similarly named in tensor board) indicating what is quantified. 

1) "best_checkpoint_epochs":

        (tag=training_set)
    - **training_set_loss.xvg** - The training loss for each individual batch, recorded after the forward pass, but before the optimizer update. The step is the global batch index, offset by step0 when resuming training.

        (tag=validation_int)
    -   **validation_int_avg_val_loss.xvg** - The average validation loss across the entire validation dataset, calculated at each validation interval. This value is used to determine whether the current model is the best checkpoint so far.
    -   **validation_int_avg_train_loss_in_val_int.xvg** - The average of the individual batch losses recorded since the previous validation interval. This provides a measure of training loss over the same interval between validating checks.

        (tag=model_checkpoint) 
    
    -   **model_checkpoints_tloss.xvg** - The training loss of the batch that triggered a new best checkpoint, recorded when the validation loss improves.

    - **model_checkpoints_val_loss_avg.xvg** - The best validation loss observed so far, updated whenever a new best checkpoint is saved. 

    - **model_checkpoints_running_tloss.xvg** - The average training loss over the batches since the previous validation interval, recorded when a new best checkpoint is saved.

        (tag=epoch_stats)

    - **epoch_stats_avg_batch_loss_in_epoch.xvg** - The average of the individual batch losses over the epoch. This is calculated by dividing )the sum of batch losses by the number of training batches. 

    In addition to these scalar logs, the routine saves model checkpoints wheneber the validation loss improves and saves a final checkpoint at the end of training (whether the validation improved or not.). 

    Checkpoint metadata includes the validation loss, training loss, training routine name, and path to the training settings. 

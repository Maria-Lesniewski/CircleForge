# @author Maria Lesniewski
# @file trainer.py
# @desc This is a lib of training epoch routines used for building models
# @desc This is a trainer class to bundle different epoch styles into one module without excessive arg passing in main scripts

from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
from  utils import model_logger as ml
from torch.utils.data import DataLoader

class Trainer:
    _epoch_registry: Dict[str, Any] = {
            # epoch registry filled by routine definitions below
            }
    def register_epoch(epoch_routine_name: str, registry=_epoch_registry):
        def decorator(epoch_function):
            """
            decorator factory for adding to dictionary of implemented epoch routines
            """
            registry[epoch_routine_name] = epoch_function
            return epoch_function
        return decorator



    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, logger, epoch_style : str):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.model : nn.Module = model.to(self.device)
        self.optimizer : optim.Optimizer = optimizer
        self.loss_fn : nn.Module = loss_fn
        self.train_loader : DataLoader = train_loader
        self.val_loader : DataLoader = val_loader
        self.logger : ml.Logger = logger
        self.epoch_routine = self._get_epoch_routine(epoch_style)

    def _get_epoch_routine(self,epoch_style):
        """
        Checks for a valid epoch call in the trainer init
        """
        if epoch_style not in self._epoch_registry:
            raise ValueError(f"Unknown training epoch style: '{epoch_style}'")
        return self._epoch_registry[epoch_style]

    def train(self, epoch_config : Dict[str, Any]):
        """
        calls the epoch routine specified by the class and trains for n_epochs
        """
        self.epoch_routine(self,**epoch_config)

    @register_epoch("best_checkpoint_epochs")
    def best_checkpoint_epochs(self, n_epochs : int , validation_interval : int):
        """
        Minimizes loss function passed according to optimizer passed
        saves models from batches wherein the validation data was 
        """

        print("BEST CHECKPOINT EPOCHS LAUNCHED")
        best_validation_loss = 10**30
        n_batches = len(self.train_loader)
        if validation_interval > n_batches:
            raise ValueError(f"validation_interval : {validation_interval} > n_batches per epoch: {n_batches}")

        for epoch in range(n_epochs):
            epoch_tloss = 0
            running_valint_tloss = 0
            for i, data in enumerate(self.train_loader):
                # Obtain batch samples
                input_img, label = data
                input_img = input_img.to(self.device)
                label = label.to(self.device)
                # zero gradient
                self.optimizer.zero_grad()
                #Eval Model
                output = self.model(input_img)
                # Back propagate
                loss = self.loss_fn(output, label)
                loss.backward()
                self.optimizer.step()

                # save loss each batch
                batch_id = (epoch*len(self.train_loader) + i)
                print(batch_id)
                self.logger.add_scalars(tag="training_set", scalars={"loss" : loss.item() }, step=batch_id)
                # append loss to runner
                running_valint_tloss += loss.item()
                epoch_tloss += loss.item()

                # Check validation set
                if i % validation_interval == validation_interval - 1:
                    avg_val_loss = self.evaluate_model()
                    avg_tloss_in_valint = running_valint_tloss / validation_interval
                    print("#### EVALUATION ####")
                    print(f"epoch:{epoch}, batch: {i}")
                    print(f"average training loss over {validation_interval} batches:")
                    print(f"{avg_tloss_in_valint}")
                    print(f"average validation loss : {avg_val_loss}")
                    self.logger.add_scalars(
                            tag="validation_int",
                            scalars={
                                "avg_val_loss" : avg_val_loss,
                                "avg_train_loss_in_val_int" : avg_tloss_in_valint
                                }, 
                            step=batch_id)

                    if avg_val_loss < best_validation_loss: 
                        print("NEW BEST MODEL")
                        best_validation_loss = avg_val_loss
                        self.logger.save_model(
                                model_id=f"{self.model.__class__.__name__}", 
                                step=epoch*len(self.train_loader) + i, 
                                model=self.model, 
                                stats={
                                    "validation_loss" : avg_val_loss, 
                                    "training_loss" : avg_tloss_in_valint},
                                metadata=True
                                )
                        self.logger.add_scalars(
                                tag="model_checkpoints",
                                scalars={
                                    "tloss" : loss.item(),
                                    "val_loss_avg" : best_validation_loss,
                                    "running_tloss" : avg_tloss_in_valint
                                },
                                step=batch_id) 
                    running_valint_tloss = 0
            print(f"----End of Epoch {epoch}----")
            print(f"EPOCH: {epoch}, average loss {epoch_tloss/n_batches}")
            epoch_tloss = 0


    def evaluate_model(self):
        """
        Evaluates the model passed over the validation set
        returns average validation loss 
        """
        self.model.eval() ## TURN OFF TRAINING MODE
        with torch.no_grad():
            running_vloss = 0
            for i, val_data in enumerate(self.val_loader):
                vinputs, vlabels = val_data
                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)
                voutputs = self.model(vinputs)
                vloss = self.loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
            avg_vloss = running_vloss / len(self.val_loader)
        self.model.train() ## TURN ON TRAINING MODE 
        return avg_vloss

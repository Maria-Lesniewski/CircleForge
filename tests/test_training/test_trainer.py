import pytest
import torch 
import torch.nn as nn
import torch.optim as optim
from test_utils.spy_logger import SpyLogger
from training.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset



## SET UP ### 
## Making these tiny pieces fixtures to use the stronger "is" check
@pytest.fixture
def model():
    return nn.Linear(2,1)

@pytest.fixture
def optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01)

@pytest.fixture
def loss_fn():
    return nn.MSELoss()

@pytest.fixture
def logger():
    return SpyLogger("./")

@pytest.fixture
def trainer(model, optimizer, loss_fn, logger):
    # Dummy training data
    x = torch.tensor([
        [0,1],
        [0,2],
        [0,3],
        [0,4]
        ], dtype=torch.float32)
    y = torch.tensor([
        [1],
        [2],
        [3],
        [4],
                ], dtype=torch.float32)
    dataset = TensorDataset(x,y)
    tr_loader = DataLoader(dataset, batch_size=1)
    
    x = torch.tensor([[0,5],])
    y = torch.tensor([[5]])
    dataset = TensorDataset(x,y)
    val_loader = DataLoader(dataset, batch_size=1)

    return Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=tr_loader,
            val_loader=val_loader,
            logger=logger,
            epoch_style="best_checkpoint_epochs"
            )

### TESTS ### 

def test_init(trainer, model, loss_fn, optimizer, logger):
    assert trainer.model is model
    assert trainer.optimizer is optimizer
    assert trainer.loss_fn is loss_fn
    assert len(trainer.train_loader) == 4
    assert len(trainer.val_loader) == 1

    for i, ele in enumerate(trainer.train_loader):
        x, y = ele
        assert x[0][0].item() == 0
        assert x[0][1].item() == y[0][0].item()
    
    x,y = next(iter(trainer.val_loader))
    assert x[0][0].item() == 0
    assert x[0][1].item() == 5
    assert y[0][0].item() == 5

    assert trainer.logger is logger

    # This is a little weird as "unit" test 
    # in that if _get_epoch_routine works in a valid key scenario this will too
    assert trainer.epoch_routine is trainer._epoch_registry["best_checkpoint_epochs"]


def test_get_epoch_routine_error(trainer):
    """ Checks for correct error from _get_epoch_routine """

    try:
        trainer._get_epoch_routine("wing_it")
        assert False
    except ValueError:
        assert True

def test_trainer_register(trainer):
    """ Checks Trainer reg mech """

    assert "perfect_routine" not in trainer._epoch_registry
    
    @Trainer.register_epoch("perfect_routine")
    def dummy_routine():
        pass

    assert "perfect_routine" in trainer._epoch_registry


def test_logging_in_train_routine(trainer):
    """ verifies logger saved metadata for training routing""" 
    trainer.train({"n_epochs" : 0, "validation_interval" : 0})
    assert trainer.logger.write_json_calls["training_settings.json"]["dictionary"] == vars(trainer)


## Other halves of trainer.train -> routines
def test_best_checkpoints_epoch_raises_error(trainer):
    try:
        trainer.train({"n_epochs": 1, "validation_interval" : 100})
        assert False
    
    except ValueError:
        assert True

def test_best_checkpoints_epoch_train_logging(trainer):

     trainer.evaluate_model = lambda: 1.0
     trainer.train({"n_epochs": 1, "validation_interval": 4})

     # Check total number of calls
     assert len(trainer.logger.add_scalars_calls) == 6

     # Check training loss was logged for every batch
     training_set_calls = { key : value for key, value in trainer.logger.add_scalars_calls.items() if key.startswith("training_set")}
     assert len(training_set_calls) == 4

     # Check validation loss was logged once, model_checkpoints logged
     assert len(trainer.logger.add_scalars_calls["validation_int"]["scalars"]) == 2
     assert len(trainer.logger.add_scalars_calls["model_checkpoints"]["scalars"]) == 3

     # Check the best model was savee
     assert len(trainer.logger.save_model_calls) == 1

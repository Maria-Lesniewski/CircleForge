import utils.io as io
import torch
import torch.nn as nn
import Models.model_options as mo
from DataSet import DoodleData as dd
from utils.model_logger import Logger
import numpy as np

# This is slower, but it should require less memory 
## Prelims for looping 
input_transforms = {
        "BasicLinear" : (lambda x : x), # Layer 1 of this model takes the image exactly
        "RotatedLinear" : (lambda x: 
            1/4*(
                x 
                +torch.rot90(x, 1, dims=(0,1)) 
                + torch.rot90(x, 2, dims=(0,1)) 
                + torch.rot90(x,3, dims=(0,1))
                )
            ), # Layer 1 of this model averages rotated reps of the image,
        "Constant" : lambda x : torch.tensor([1]) # This model uses no features. Just 1s
        }

augmenting_functions = {
        "BasicLinear" : (
            lambda X : torch.cat(
                [X, torch.ones((X.shape[0], 1), dtype=X.dtype)],dim=1)
            ),
        "RotatedLinear" : (
            lambda X : torch.cat(
                [X, torch.ones((X.shape[0], 1), dtype=X.dtype)],
                dim=1)
            ),
        "Constant" : lambda X : X, 
        }
ele = io.multiple_choice_q("This script will use linear least squares to train a model", [item for item in input_transforms.keys()])
data_fp = io.artifact_path("final_data/train_val_test_split_annotations.csv")
train_data = dd.CircleData(
        csv_file=data_fp,
        target_columns=["ncircles"],
        split_tag="train"
        )
# One logger for the experiment
logger = Logger(io.find_here() / "Results")

print(f"####### {ele} ######")
print("Unpacking dataset")
Y = []
X = []
for features, label in train_data: # Note because this is samplewise, we really want xW = y 
    if (ele == "Constant"):
        x = input_transforms[ele](features).numpy()
    else:
        x =(input_transforms[ele](features)).detach().flatten().numpy()
    X.append(x)
    Y.append(label.detach().item())
Y = torch.tensor(Y, dtype=torch.float64).detach()
X = torch.tensor(X, dtype=torch.float64).detach()

print("Preparing to solve linear least squares")
# Augment the feature column with a 1 across samples. (coeff will be b)
aug_X = augmenting_functions[ele](X)

print(f"Solving LLS for {ele}")
S = torch.linalg.lstsq(aug_X,Y,rcond=None, driver="gelsd")
W = S.solution
#W, _, _, _ = np.linalg.lstsq(aug_X,Y,rcond=None)

print(f"Loading to Torch")
model = mo.ModelFactory.create(ele,{})
torch.nn.utils.vector_to_parameters(
        torch.tensor(W, dtype=torch.float64),
        model.parameters()
        )

print("Evaluating Solutions on Least Squares Objective")
print(f"The average is {torch.mean(Y)}")
if (ele == "Constant"):
    item = [ele.item() for ele in model.parameters()][0]
    print(f"Constant got params {item}")

loss_crit = nn.MSELoss()
# Check with basic linalg
y_hat = aug_X @ W
del(aug_X)
if (ele != "Constant"):
    yt_hat = (model(X.unsqueeze(0).reshape(-1,50,50))).squeeze(-1)
else:
    yt_hat = (model(X.unsqueeze(0)).squeeze(-1))

print(yt_hat.shape)
print(y_hat.shape)
print(Y.shape)
tloss = loss_crit(yt_hat, Y)
loss = loss_crit(y_hat, Y)

print(f"{ele} : torch Loss {tloss}")
print(f"{ele} : linalg Loss {loss}")
meta_data = {"train_loss" : tloss.item(), "train_mse" : torch.sqrt(tloss).item()}
if (ele == "Constant"):
    meta_data["coefficient"] = item
logger.save_model(model_id=ele, step=0, model=model, stats=meta_data, metadata=True)

logger.close()


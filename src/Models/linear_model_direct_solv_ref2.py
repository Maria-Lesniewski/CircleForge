import numpy as np
import DataSet.DoodleData as dd
import torch
import Modeling.model_options as mo
import torch.nn as nn
from torch.nn.utils import vector_to_parameters
### Data sets
train_data = dd.CircleData(
    csv_file="DataSet/samples/split_annotations.csv",
    target_columns=[
        "ncircles",
    ],
    split_tag ="train"
)
val_data = dd.CircleData(
    csv_file="DataSet/samples/split_annotations.csv",
    target_columns=[
        "ncircles",
    ],
    split_tag ="val"
)
test_data = dd.CircleData(
    csv_file="DataSet/samples/split_annotations.csv",
    target_columns=[
        "ncircles",
    ],
    split_tag ="test"
)

## Model 
linear = mo.RotatedLinear()
loss_crit = nn.MSELoss()

## Lets directly solve this with all the samples if we can

X = []
y = []

for features, label in train_data:
    x = features
    x = 1/4*(x + torch.rot90(x, 1, dims=(0,1)) + torch.rot90(x, 2, dims=(0,1)) + torch.rot90(x,3, dims=(0,1)))
    X.append(x.detach().cpu().flatten().numpy())
    #X.append(features.detach().cpu().flatten().numpy())
    # detach - no need to track gradients
    # cpu - keeps on cpu (We built our models locally because we dont have a gpu)
    # numpy - convert to numpy for matrix manipulation
    # flatten() - Flatten the vector into dimx x dimy x dimz (dimz = 1)
    y.append(label.detach())

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)
aug_X_for_b = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # Target Shape: (29500, 2501)
#np.column_stack((X, np.ones_like(X)))  # Append 1 to every sample for bias param

print(f"shape {aug_X_for_b.shape}")
# Soln,squared residuals, rank, singular values
W, residuals, rank, singular_values = np.linalg.lstsq(aug_X_for_b,y, rcond=None)

print(f"rank: {rank}")
print(f"len: {len(X)}")

# Check with basic linalg
y_pred = aug_X_for_b @ W

mse = np.mean((y_pred - y) ** 2)
rmse = np.sqrt(mse)

print("Training MSE:", mse)
print("Training RMSE:", rmse)

# Put into torch model 
linear = mo.RotatedLinear()

vector_to_parameters(torch.tensor(W, dtype=torch.float32), linear.parameters())

fX = []
# Need images directly for torch 
for features, label in train_data:
    fX.append(features)

fX = torch.stack(fX)
output = linear(torch.tensor(fX, dtype=torch.float32))
loss = loss_crit(output,torch.tensor(y, dtype=torch.float32))

print("From torch framework:")
print("Training MSE:", loss.detach().numpy())
print("Training RMSE:", np.sqrt(loss.detach().numpy()))
model_path = "Final_Least_Squares_rotated_opt_model"
torch.save(linear.state_dict(), model_path)


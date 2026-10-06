import numpy as np
import DataSet.DoodleData as dd
import torch
import Models.model_options as mo
import torch.nn as nn
import utils.io as io
from torch.nn.utils import vector_to_parameters
### Data sets
print("Loading training/validation data samples")
data_fp = io.artifact_path("final_data/train_val_test_split_annotations.csv")
train_data = dd.CircleData(
        csv_file=data_fp,
        target_columns=["ncircles"],
        split_tag="train"
        )
val_data = dd.CircleData(
        csv_file=data_fp,
        target_columns=["ncircles"],
        split_tag="val"
        )

## Model 
linear = mo.BasicLinear()
loss_crit = nn.MSELoss()

## Lets directly solve this with all the samples if we can

X = []
y = []

for features, label in train_data:
    X.append(features.detach().cpu().flatten().numpy())
    # detach - no need to track gradients
    # cpu - keeps on cpu (We built our models locally because we dont have a gpu)
    # numpy - convert to numpy for matrix manipulation
    # flatten() - Flatten the vector into dimx x dimy x dimz (dimz = 1)
    y.append(label.detach())

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)

print("SCRIPT 1")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X[0, :10]:", X[0, :10])
print("y[:10]:", y[:10])
print("X sum:", X.sum())
print("y sum:", y.sum())


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
linear = mo.BasicLinear()

vector_to_parameters(torch.tensor(W, dtype=torch.float32), linear.parameters())

output = linear(torch.tensor(X, dtype=torch.float32))
loss = loss_crit(output,torch.tensor(y, dtype=torch.float32))

print("From torch framework:")
print("Training MSE:", loss.detach().numpy())
print("Training RMSE:", np.sqrt(loss.detach().numpy()))
model_path = "Final_Least_Squares_opt_model"
torch.save(linear.state_dict(), model_path)


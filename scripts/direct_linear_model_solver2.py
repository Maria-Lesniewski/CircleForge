import utils.io as io
import torch
import torch.nn as nn
import models.model_options as mo
from dataset.circle_data import CircleData
from utils.model_logger import Logger
import numpy as np

# This is sort of a one off application for least squares with supported models
# and one target regression 

## HARD CODED OPTIONS (linear models only)
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

# OBTAIN CONFIG
args = io.parse_train_args()
if (args.config is None):
    raise ValueError("CL requires config, will prompt for model choice if not provided")
choice = args.model
config = io.load_config_from_CL(args.config)

if (choice is None):
    choice = io.multiple_choice_q(
            "This script will use linear least squares to train a model",
            [item for item in input_transforms.keys()]
            )

### UNPACK CONFIG
write_path = io.artifact_path(config["training"]["logging_path"])
data_fp = io.artifact_path(config["data"]["annotations_file"])
tfunc = config["data"]["feature_transform"]
col = config["data"]["target_columns"]
if len(col) > 1:
    raise ValueError("target_columns for least squares regression must be len 1")


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

train_data = CircleData(
        csv_file=data_fp,
        target_columns=col,
        split_tag="train",
        transform=tfunc
        )
# One logger for the experiment
experiment_path = io.make_artifact_dir(write_path)
logger = Logger( experiment_path / "LeastSquaresResults")
choice = str(choice)
print(f"####### {choice} ######")
print("Unpacking dataset")
Y = []
X = []
for features, label in train_data: # Note because this is samplewise, we really want xW = y 
    if (choice == "Constant"):
        x = input_transforms[choice](features).numpy()
    else:
        x =(input_transforms[choice](features)).detach().flatten().numpy()
    X.append(x)
    Y.append(label.detach().item())
Y = torch.tensor(Y, dtype=torch.float64).detach()
X = torch.tensor(X, dtype=torch.float64).detach()

print("Preparing to solve linear least squares")
# Augment the feature column with a 1 across samples. (coeff will be b)
aug_X = augmenting_functions[choice](X)

print("Finding max eigenvalue of the MSE Hessian")

n = aug_X.shape[0]

### SPECTRAL ANALYSIS AND SOLUTION BLOCK
spectral_analysis = {"Model" : choice}
spectral_analysis["torch.linalg.svd"] = {}
sigma_max = torch.linalg.svdvals(aug_X).max()
sigma_min = torch.linalg.svdvals(aug_X).min()
lambda_max = 2.0 * sigma_max.square() / n
condition_number = sigma_max / sigma_min
spectral_analysis["torch.linalg.svd"]["largest_singular_value"] = sigma_max.item()
spectral_analysis["torch.linalg.svd"]["smallest_singular_value"] = sigma_min.item()
spectral_analysis["torch.linalg.svd"]["largest_hessian_eigenvalue"] = lambda_max.item()
spectral_analysis["torch.linalg.svd"]["condition_number"]  = condition_number.item()
eta_max = 2.0 / lambda_max
spectral_analysis["optimization"] = {}
spectral_analysis["optimization"]["maximum_theoretical_learning_rate"] = eta_max.item()
print(f"Model: {choice}")
print(f"Largest singular value: {sigma_max.item():.6e}")
print(f"Smallest singular value: {sigma_min.item():.6e}")
print(f"Condition number: {condition_number.item():.6e}")
print(f"Largest Hessian eigenvalue: {lambda_max.item():.6e}")
print(f"Maximum theoretical learning rate (Basic Gradient Descent): {eta_max.item():.6e}")



print(f"Solving LLS for {choice}")
S = torch.linalg.lstsq(aug_X,Y,rcond=None, driver="gelsd")
W = S.solution
print(f"Solved least squares with rank: {S.rank}")
spectral_analysis["torch.linalg.lstq"] = {}
spectral_analysis["torch.linalg.lstq"]["condition_number"] = (S.singular_values.max() / S.singular_values.min()).item()
spectral_analysis["torch.linalg.lstq"]["max_value"] = S.singular_values.max().item()
spectral_analysis["torch.linalg.lstq"]["min_value"] = S.singular_values.min().item()
spectral_analysis["torch.linalg.lstq"]["rank"] = S.rank.item()
logger.write_json("spectral_analysis.json", spectral_analysis)


#W, _, _, _ = np.linalg.lstsq(aug_X,Y,rcond=None)
#### INSERTED SANITY CHECK ####

print(f"Loading to Torch")
model = mo.ModelFactory.create(choice,{})
torch.nn.utils.vector_to_parameters(
        torch.tensor(W, dtype=torch.float32),
        model.parameters()
        )

print("Evaluating Solutions on Least Squares Objective")
print(f"The average is {torch.mean(Y)}")
if (choice == "Constant"):
    item = [choice.item() for choice in model.parameters()][0]
    print(f"Constant got params {item}")

loss_crit = nn.MSELoss()
# Check with basic linalg
y_hat = aug_X.to(dtype=torch.float32) @ W.to(dtype=torch.float32)
X = X.to(dtype=torch.float32)
if (choice != "Constant"):
    yt_hat = (model(X.unsqueeze(0).reshape(-1,50,50))).squeeze(-1)
else:
    yt_hat = (model(X.unsqueeze(0)).squeeze(-1))

print(yt_hat.shape)
print(y_hat.shape)
print(Y.shape)
tloss = loss_crit(yt_hat, Y)
loss = loss_crit(y_hat, Y)






# Evaluate the same solution in multiple ways

W64 = W
W32 = W.float()

# 1. Float64 direct evaluation
pred64 = aug_X @ W64

# 2. Float32 augmented matrix evaluation
pred32_aug = aug_X.float() @ W32

# 3. Float32 model evaluation
pred32_model = yt_hat

# Compare predictions
print("Float64 vs Float32 augmented:",
      (pred64 - pred32_aug.double()).abs().max().item())

print("Float32 augmented vs model:",
      (pred32_aug - pred32_model).abs().max().item())

# Compare RMSE
for name, pred in [
    ("Float64", pred64),
    ("Float32 augmented", pred32_aug),
    ("Float32 model", pred32_model)
]:
    rmse = torch.sqrt(torch.mean((pred.double() - Y)**2))
    print(f"{name} RMSE: {rmse.item():.12e}")


###############################
# Separate weight multiplication from bias addition

weights = model.f1.weight.detach()
bias = model.f1.bias.detach()

manual = X @ weights.T

manual_with_bias = manual + bias

print("Bias:", bias.item())

print(
    "Manual vs augmented:",
    (manual_with_bias.squeeze() - y_hat).abs().max().item()
)

print(
    "Manual vs model:",
    (manual_with_bias.squeeze() - yt_hat).abs().max().item()
)

print(
    "Bias contribution:",
    bias.item()
)
###################################################






print(f"{choice} : torch Loss {tloss}")
print(f"{choice} : linalg Loss {loss}")
meta_data = {"train_loss" : tloss.item(), "train_mse" : torch.sqrt(tloss).item()}
if (choice == "Constant"):
    meta_data["coefficient"] = item
logger.save_model(model_id=choice, step=0, model=model, stats=meta_data, metadata=True)

logger.close()


import DataSet.DoodleData as dd
from pathlib import Path
import torch
import torch.nn as nn
import os
import Modeling.model_options as mo
from torch.utils.data import DataLoader
import numpy as np
## Load your best model parameters here 
#best_params = input("Enter name of best fit model file")
#best_params = "Final_Least_Squares_opt_model"
best_params = "Final_Least_Squares_rotated_opt_model"
log_str = "analysis_log_model_" + best_params
best_model_path = Path(best_params)
# Log what you're doing since we have CL input
with open(log_str, 'w', encoding='utf-8') as logfile:
    logfile.write("Last model analyzed was" + best_params + "\n")

#linear_model = mo.BasicLinear()
linear_model = mo.RotatedLinear()
linear_model.load_state_dict(torch.load(best_model_path))
linear_model.eval() # No gradients here, just final model diagnostics 
loss = nn.MSELoss()

# Load the Different Data Sets and all their labels
data_dict = {"train" : 0, "val" : 1, "test" : 2}
labels_map = {
        "ncircles" : 0,
        "nbig" : 1,
        "nsmall" : 2 ,
        "area_of_circles" : 3,
        "area_of_negative" : 4,
        "max_circle_units" : 5,
        "fill_fraction" : 6
        }
datas = []
for value in data_dict:
    datas.append(dd.CircleData(
        csv_file="DataSet/samples/split_annotations.csv",
        target_columns=[
        "ncircles",
        "nbig",
        "nsmall",
        "area_of_circles",
        "area_of_negative",
        "max_circle_units",
        "fill_fraction"
            ],
        split_tag=value
        )
        )

# Predict and analyze splits
for i, (sets,key) in enumerate(zip(datas, data_dict)):
    # Create a Directory for data 
    save_dir = Path("Diagnostics/" + key)
    
    # Compare to a mean predictor
    N = torch.stack([sets[i][1][labels_map["ncircles"]] for i in range(len(sets))])
    mean_predictor = N.mean().unsqueeze(dim=0).unsqueeze(dim=1)

    if not os.path.exists(Path(save_dir)):
        os.mkdir(save_dir)
    # Make predictions from model
    loader = DataLoader(datas[i], batch_size=1)
    error = []
    mperror = []
    predict_v_true = []
    error_v_true = []
    for j,data in enumerate(loader, 0):
        # Evaluate error per sample so that we can calc distro data if we want later
        with torch.no_grad():
            img, labels = data
            predictions = linear_model(img)
            ncircleslabel = labels[:,labels_map["ncircles"]].unsqueeze(1)
            error.append(loss(predictions, ncircleslabel).item())
            error_v_true.append([ncircleslabel.squeeze(1).item(), loss(predictions, ncircleslabel).item()])
            mperror.append(loss(mean_predictor, ncircleslabel).item())
            predict_v_true.append([predictions.squeeze(), ncircleslabel.squeeze()])
    R2 = 1 - sum(error)/sum(mperror)
    mask = (np.abs(error) <= 1).astype(int)
    acc = sum(mask)/len(mask)
    np.savetxt(save_dir / "model_MSE_per_sample.xvg", np.column_stack((np.arange(0,len(datas[i])), error)))
    print(mperror[0])
    np.savetxt(save_dir / "mean_predictor_MSE_per_sample.xvg" , np.column_stack((np.arange(0, len(datas[i])), mperror)))
    np.savetxt(save_dir / "Error_v_True_value.xvg", error_v_true)
    np.savetxt(save_dir / "Predict_v_True_value.xvg", predict_v_true)
    np.savetxt(save_dir / "R2_accuracy.xvg", np.array([["R2 = ", R2], ["Accuracty = ", acc]]), fmt="%s")
    

## Plot 
# y_pred vs y_actual
# y_pred vs. avg y 

# Look at error binned by labels 

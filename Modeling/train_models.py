import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import DataSet.DoodleData as dd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Modeling.model_options as mo


### Data 
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
## Loaders
trloader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=4
        )
val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=True,
        num_workers=4
        ) 
testloader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=True,
        num_workers=4
        )

writer = SummaryWriter('log/training')

### Model 
linear = mo.BasicLinear()
loss_crit = nn.MSELoss() # This is the square of RMS and is easier for gradients
optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)

old_school_log = []
model_log = []
best_vloss = 1000000000
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trloader, 0):
        # Obtain 32 samples/labels from loader
        input_img, label = data

        # zero the gradient parameters
        optimizer.zero_grad()

        # Eval Model
        output = linear(input_img)

        # Back propagate loss
        loss = loss_crit(output,label)
        loss.backward()
        optimizer.step()

        # print stats
        running_loss += loss.item()
        if i % 80 == 79:
            print(f"epoch: {epoch} avg loss {running_loss/80}")
            
            running_vloss = 0
            linear.train(False)
            for j, vdata in enumerate(val_loader,0):
                vinputs, vlabels = vdata
                voutputs = linear(vinputs)
                vloss = loss_crit(voutputs, vlabels)
                running_vloss += vloss.item()
            linear.train(True)
            
            avg_vloss = running_vloss / len(val_loader)
            avg_loss = running_loss / 80

            #Log
            writer.add_scalars('loss vs. batch no.', 
                    { 'Training' : avg_loss, 'Validation' : avg_vloss, 'Epoch #' : epoch }, i*epoch
                    )
            old_school_log.append([i*epoch, avg_loss, avg_vloss] )
            running_loss = 0.00

            if avg_vloss < best_vloss:
                print(f"New best model {epoch} {i}")
                best_vloss = avg_vloss
                model_path = 'CachedModels/model_{}_{}'.format(epoch, i)
                torch.save(linear.state_dict(), model_path)
                model_log.append([str(model_path), best_vloss])

print('Finished Training')


writer.flush()
np.savetxt("log/loss_epoch_vloss_vs.train_batch_ID.xvg", old_school_log)
np.savetxt("CachedModels/model_dict_and_validation_loss.txt", model_log, fmt="%s")

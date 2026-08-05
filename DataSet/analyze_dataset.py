import DoodleData as dd 
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import pandas as pd 
dataset = dd.CircleData(
    csv_file="data_folder/annotations.csv",
    target_columns=[
        "ncircles",
        "nbig",
        "nsmall",
        "area_of_circles",
        "area_of_negative",
        "max_circle_units"
    ]
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

for x, y in loader:
    print(x.shape, y.shape)
    break
labels_map = {
        "ncircles" : 0,
        "nbig" : 1,
        "nsmall" : 2 ,
        "area_of_circles" : 3,
        "area_of_negative" : 4,
        "max_circle_units" : 5
        }

cols, rows = 3,3
figure = plt.figure(figsize=(cols*rows-1, cols*rows-1))
for i in range(1, cols*rows +1):
    sample_idx = torch.randint(len(dataset), size=(1,)).item()
    img, label = dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    print(label)
    plt.title(label[labels_map["ncircles"]].item())
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# collect labels from dataset
records = []

for i in range(len(dataset)):
    _, label = dataset[i]

    records.append({
        "ncircles": label[labels_map["ncircles"]].item(),
        "nbig": label[labels_map["nbig"]].item(),
        "nsmall": label[labels_map["nsmall"]].item(),
        "area_of_circles": label[labels_map["area_of_circles"]].item(),
        "area_of_negative": label[labels_map["area_of_negative"]].item(),
        "max_circle_units" : label[labels_map["max_circle_units"]].item()
    })

analysis_df = pd.DataFrame(records)

print(analysis_df.head())

groups = analysis_df.groupby("max_circle_units")

def hist_prop(propert, groups):
    # Take first 9 groups
    first_nine = list(groups)[:9]
    # Create 3x3 grid
    fig, axes = plt.subplots(3,3,figsize=(12, 10))

    # Flatten axes so we can iterate easily
    axes = axes.flatten()

    for ax, (nmax, group) in zip(axes, first_nine):

        ax.hist(
            group[propert]/(600*600),
            bins=20,
            edgecolor="black"
        )

        ax.set_title(f"NMax = {nmax}")
        ax.set_xlabel(propert)
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

#hist_prop("area_of_negative", groups)

x = list(groups)[:9]
y = groups["area_of_negative"].mean()[:9]/(600*600)

plt.figure(figsize=(8,5))

for nmax, group in list(analysis_df.groupby("max_circle_units")):
    plt.scatter(
        [nmax], #* len(group),
        group["area_of_negative"].mean(),
        alpha=0.4,
        label=f"NMax {nmax}"
    )

plt.xlabel("NMax")
plt.ylabel("Number of circles")
plt.legend()
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
import os
from tqdm import tqdm
 
import warnings
warnings.filterwarnings('ignore')

from models import *
from config import *
from clip import *
from dataset import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "Using device: ",
    device,
    f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
)
 
model = CLIP(
        emb_dim,
        vit_layers,
        vit_d_model,
        img_size,
        patch_size,
        n_channels,
        vit_heads,
        vocab_size,
        max_seq_length,
        text_heads,
        text_layers,
        text_d_model,
        retrieval=False,
    ).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
 
total_params = 0
total_params = sum(
    [param.numel() for param in model.parameters() if param.requires_grad]
)
 
print(
    f"Total number of trainable parameters: {total_params}; i.e., {total_params/1000000:.2f} M"
)
 
# Total number of trainable parameters: 532641; i.e., 0.53 M


# Load the dataset
df = pd.read_csv('fashion/myntradataset/styles.csv', usecols=['id',  'subCategory'])
 
unique, counts = np.unique(df["subCategory"].tolist(), return_counts = True)
print(f"Classes: {unique}: {counts}")
 
# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.10, random_state=42)
 
# Print the sizes of the datasets
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
class_names = df['subCategory'].unique()
class_names = [str(name).lower() for name in class_names]
 
# Replace in-place
for i, name in enumerate(class_names):
    if name == "lips":
        class_names[i] = "lipstick"
    elif name == "eyes":
        class_names[i] = "eyelash"
    elif name == "nails":
        class_names[i] = "nail polish"
 
captions = {idx: class_name for idx, class_name in enumerate(class_names)}
 
for idx, caption in captions.items():
    print(f"{idx}: {caption}\n")  





train_dataset = MyntraDataset(data_frame=train_df, captions=captions, target_size=80)
val_dataset = MyntraDataset(data_frame=val_df, captions=captions, target_size=80)
test_dataset = MyntraDataset(data_frame=val_df, captions=captions, target_size=224)
 
print("Number of Samples in Train Dataset:", len(train_dataset))
print("Number of Samples in Validation Dataset:", len(val_dataset))
 
# Number of Samples in Train Dataset: 38360
# Number of Samples in Validation Dataset: 4278
 
train_loader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size, num_workers=5
)
val_loader = DataLoader(
    val_dataset, shuffle=False, batch_size=batch_size, num_workers=5
)
test_loader = DataLoader(
    test_dataset, shuffle=False, batch_size=batch_size, num_workers=5
)
 
# Sanity check of dataloader initialization
len(next(iter(train_loader)))  # (img_tensor,label_tensor)



best_loss = np.inf
for epoch in range(epochs):
    epoch_loss = 0.0  # To accumulate the loss over the epoch
    with tqdm(
        enumerate(train_loader, 0),
        total=len(train_loader),
        desc=f"Epoch [{epoch+1}/{epochs}]",
    ) as tepoch:
        for i, data in tepoch:
            img, cap, mask = (
                data["image"].to(device),
                data["caption"].to(device),
                data["mask"].to(device),
            )
            optimizer.zero_grad()
            loss = model(img, cap, mask)
            loss.backward()
            optimizer.step()
 
            # Update the progress bar with the current loss
            tepoch.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
 
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.3f}")
 
    # Save model if it performed better than the previous best
    if avg_loss <= best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "clip.pt")
        print("Model Saved.")




model = CLIP(
    emb_dim,
    vit_layers,
    vit_d_model,
    img_size,
    patch_size,
    n_channels,
    vit_heads,
    vocab_size,
    max_seq_length,
    text_heads,
    text_layers,
    text_d_model,
    retrieval=False,
).to(device)
 
model.load_state_dict(torch.load("clip.pt", map_location=device))
 
# print([x for x in val_dataset.captions.values()])
# Getting dataset captions to compare images to
text = torch.stack([tokenizer(x)[0] for x in val_dataset.captions.values()]).to(device)
# print(text)
mask = torch.stack([tokenizer(x)[1] for x in val_dataset.captions.values()])
mask = (
    mask.repeat(1, len(mask[0]))
    .reshape(len(mask), len(mask[0]), len(mask[0]))
    .to(device)
)
 
correct, total = 0, 0
with torch.no_grad():
    for data in val_loader:
 
        images, labels = data["image"].to(device), data["caption"].to(device)
        image_features = model.vision_encoder(images)
        text_features = model.text_encoder(text, mask=mask)
 
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, indices = torch.max(similarity, 1)
 
        pred = torch.stack(
            [tokenizer(val_dataset.captions[int(i)])[0] for i in indices]
        ).to(device)
        correct += int(sum(torch.sum((pred == labels), dim=1) // len(pred[0])))
        # print(pred.shape)
 
        total += len(labels)
 
print(f"\nModel Accuracy: {100 * correct // total} %")
# The tokenized ground truth caption labels and predicted labels were compared, and we obtained 85% accuracy on the validation dataset.

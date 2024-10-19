"""
Converts a list of (writer, [list of (file,class)]) into torch.tensor,
For each writer, creates a `.pt` file containing `data` and `targets`,
The resulting file is saved in `intermediate/data_as_tensor_by_writer'
"""
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import timm  # Import for embedding extraction
from torchvision import transforms, datasets  # Import necessary for transforms
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get model from timm
model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0).to(device)
model.requires_grad_(False)
model = model.eval()

# get the required transform function for the given feature extractor
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config)

def extract_embedding(img):
    """Extracts embedding from image using DINOv2 or similar model."""
    img = img.convert('RGB')  # Ensure RGB mode
    img = transform(img).unsqueeze(0).to(device)  # Transform and batchify image
    with torch.no_grad():
        output = model.forward_features(img)
        output = model.forward_head(output, pre_logits=True)
    return output.cpu().numpy()

def relabel_class(c):
    """ Same relabeling function as before """
    if c.isdigit() and int(c) < 40:
        return int(c) - 30
    elif int(c, 16) <= 90:  # uppercase
        return int(c, 16) - 55
    else:
        return int(c, 16) - 61

# class FEMNISTDataset(Dataset):
#     """Custom Dataset for FEMNIST based on writer and image paths."""
#     def __init__(self, data_list):
#         self.data_list = data_list

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         file_path, c = self.data_list[idx]
#         img = Image.open(file_path)
#         embedding = extract_embedding(img)
#         label = relabel_class(c)
#         return embedding, label

# def save_embeddings_by_writer(writers, save_dir, batch_size=256, num_workers=4):
#     """Processes and saves embeddings using DataLoader."""

#     for w, l in tqdm(writers):
#         dataset = FEMNISTDataset(l)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

#         data, targets = [], []
#         for embeddings, labels in dataloader:
#             data.append(embeddings)
#             targets.append(labels)

#         if data:
#             data = torch.cat(data)  # Stack embeddings
#             targets = torch.cat(targets)  # Stack labels

#             torch.save((data, targets), os.path.join(save_dir, f"{w}.pt"))

if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)  # Set the multiprocessing start method

    by_writers_dir = os.path.join('intermediate', 'images_by_writer.pkl')
    save_dir = os.path.join('intermediate', 'data_as_tensor_by_writer')

    os.makedirs(save_dir, exist_ok=True)

    with open(by_writers_dir, 'rb') as f:
        writers = pickle.load(f)

    for (w, l) in tqdm(writers):

        data = []
        targets = []

        for (f, c) in l:
            file_path = os.path.join(f)
            img = Image.open(file_path)
            embedding = extract_embedding(img)  # Extract embedding instead of raw pixel data
            nc = relabel_class(c)
            data.append(embedding)
            targets.append(nc)

        if len(data) > 2:
            data = torch.tensor(np.stack(data))  # Stack embeddings
            targets = torch.tensor(np.stack(targets))

            trgt_path = os.path.join(save_dir, w)

            torch.save((data, targets), os.path.join(save_dir, f"{w}.pt"))

    #save_embeddings_by_writer(writers, save_dir, batch_size=256, num_workers=4) # num_workers=16 in create_db.py




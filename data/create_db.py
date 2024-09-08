from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm

import torchvision.transforms as T
import torchvision
import torch

import numpy as np
import timm
import os


def get_dataloaders(args, transforms):
    """
    Get the dataloaders for the selected dataset and backbone.
    :param args: Namespace containing parameters and hyperparameters of the current run.
    :param transforms: Transform function. 
    :return data: Dictionary containing both train and test dataloaders.
    """

    #data_path = os.path.join(args.dataset_root,args.dataset)
    data_path = args.dataset_root

    # convert img to RGB & append after ToTensor() 
    #   this is used for MNIST and F-MNIST
    to_rgb = T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x) 

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms)
    elif args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms)
    elif args.dataset == "mnist":
        transforms.transforms.insert(-1, to_rgb)
        trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms)
        testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms)
    elif args.dataset == "fashion-mnist":
        transforms.transforms.insert(-1, to_rgb)
        trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transforms)
        testset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transforms)
    else:
        raise ValueError(f"{args.dataset} not available")
    
    dataloaders = {}

    trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle = False, num_workers=16, persistent_workers=True)
    testloader = DataLoader(testset, batch_size = args.batch_size, shuffle = False, num_workers=16, persistent_workers=True)

    dataloaders['train'] = trainloader
    dataloaders['test'] = testloader

    return dataloaders


def extract_embeddings(model, device, dataloader):
    """
    Make inference on the given dataset through the chosen backbone.
    :param model: Selected backbone.
    :param device: Running device.
    :param dataloader: Current dataloader (train|test)
    :return data: Dictionary containing the extracted data.
    """

    embeddings_db, labels_db = [], []

    for extracted in tqdm(dataloader):

        images, labels = extracted
        images = images.to(device)

        output = model.forward_features(images)
        output = model.forward_head(output, pre_logits=True)

        labels_db.extend(labels)
        embeddings_db.extend(output.detach().cpu().numpy())
    

    data = {
        'embeddings': embeddings_db,
        'labels': labels_db
    }

    return data



def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get model from timm
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0).to(device)
    model.requires_grad_(False)
    model = model.eval()

    # get the required transform function for the given feature extractor
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # get dataloaders
    dataloaders = get_dataloaders(args, transforms)

    # create database folder, if necessary
    os.makedirs(os.path.join(args.database_root),exist_ok=True)

    for split in ['train','test']:

        # get database of embeddings in the form
        #   db = {'embeddings' : [...], 'labels' : [...]
        db = extract_embeddings( model = model, 
                                 device = device,
                                 dataloader = dataloaders[split])
        
        # store database: database_root / dataset / train|test.npz
        np.savez(os.path.join(args.database_root,f'{split}.npz'), **db)


if __name__ == '__main__':

    parser = ArgumentParser()

    # GENERAL
    parser.add_argument('--dataset_root', type=str, default="tmp/assets/data", help='define the dataset root')
    parser.add_argument('--database_root', type=str, default="tmp/assets/database", help='define the database root')

    # DATASET & HYPERPARAMS
    parser.add_argument('--dataset', type=str, required=True, help='define the dataset name') 
    parser.add_argument('--backbone', type=str, default='vit_base_patch14_dinov2.lvd142m', help='define the feature extractor')
    parser.add_argument('--batch_size', type=int, default=128, help='define the batch size')

    # ADD METHOD
    args = parser.parse_args()

    main(args)
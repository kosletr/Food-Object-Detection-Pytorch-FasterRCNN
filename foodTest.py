# %%
from prepDataset import load_categories, set_split

old_root = "../UECFOOD100"

root = "../splitUECFood100"
train_split = 0.70
valid_split = 0.20
# test_split is set to  (1 - train_split - valid_split)


classes = load_categories(old_root)
bbox = set_split(old_root, root, train_split, valid_split)

#%%
"""
from matplotlib import pyplot as plt, patches

for i in range(10):
    
    fig, ax = plt.subplots(1)

    file_info = data.loc[i]
    name = str(file_info.img)+".jpg"
    idx = list.index(image, name)
    img_path = os.path.join(class_dir_path, image[idx])
    m = plt.imread(img_path)


    ax.imshow(m)
    rect = patches.Rectangle((file_info.x1,file_info.y1),file_info.x2-file_info.x1,file_info.y2-file_info.y1,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    plt.title(categories.loc[file_info.category][0])
    
    plt.show()
"""

# %%

import torch, os
from torch import optim, nn, functional as F
from torch.utils.data import dataloader
from torchvision import transforms, datasets

def data_loader(root, batch_size):

    sets = ['train','valid', 'test']

    dirs = { x : os.path.join(root, x) for x in sets }

    # Dictionary with the transformations to be applied in each set
    transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    }

    # Load Images
    dataset = {
        x : datasets.ImageFolder(dirs[x], transform=transform[x]) for x in sets
    }

    # Create a Data Loader with a given Batch Size
    data_loader = {
        x: dataloader.DataLoader(dataset[x], batch_size=batch_size) for x in sets
    }

    return data_loader

# %%

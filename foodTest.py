# %%
import os
import pandas as pd
from matplotlib import pyplot as plt, patches

data_dir = "UECFOOD100"
bb_file = 'bb_info.txt'
categories_file = 'category.txt'
image = []

categories = pd.read_csv(os.path.join(data_dir,categories_file), sep='\t')

for class_dir in os.scandir(data_dir):

    if class_dir.is_dir():

        class_dir_path = os.path.join(data_dir, class_dir.name)

        for file in os.scandir(class_dir_path):

            file_path = os.path.join(class_dir_path, file.name)

            if file.name == bb_file: 

                data = pd.read_csv(file_path, delim_whitespace=True)
                data['category'] = int(class_dir.name)

            elif file.name.endswith(".jpg"):

                image.append(file.name)
        
            


#%%

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


# %%

import os
import pandas as pd
from matplotlib import pyplot as plt, patches


def load_categories(data_dir, categories_file):

    # Import categories from categories.txt as a list
    categ_df = pd.read_csv(os.path.join(data_dir,categories_file), sep='\t')
    categories = list(categ_df.name)

    # Add background as category 0
    categories.insert(0, 'background')
    return categories


data_dir = "UECFOOD100"
categories_file = 'category.txt'

classes = load_categories(data_dir, categories_file)

# %%



# %%

import torch
from torch import optim, nn, functional as F
from torch.utils.data import dataloader
from torchvision import transforms, datasets

def data_loader(root, batch_size):

    train_dir = os.path.join(root, 'train')
    valid_dir = os.path.join(root, 'valid')
    test_dir = os.path.join(root, 'test')

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
        'train': datasets.ImageFolder(train_dir, transform=transform['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=transform['valid']),
        'test': datasets.ImageFolder(test_dir, transform=transform['test'])
    }

    # Create a Data Loader with a given Batch Size
    data_loader = {
        'train': dataloader.DataLoader(dataset['train'], batch_size=batch_size),
        'valid': dataloader.DataLoader(dataset['valid'], batch_size=batch_size),
        'test': dataloader.DataLoader(dataset['test'], batch_size=batch_size)  
    }

    return data_loader

# %%
data_dir = "UECFOOD100"
import os
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".jpg"):
             print(files)

# %%

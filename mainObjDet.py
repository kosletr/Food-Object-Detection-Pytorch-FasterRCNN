# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# Altered to fit the problem of Food Classification and Object Detection
# for the UECFOOD100 Dataset

# !!! Numpy 1.17 must be installed !!!

# %%

import os
# import numpy as np
import torch
from PIL import Image
# import pandas as pd

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from prepDataset import load_categories, merge_info, convert_dataset

# %%


class foodDataset(object):
    def __init__(self, root, transforms, bbox_info):
        self.root = root
        self.transforms = transforms
        self.bbox_info = bbox_info

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.imgs[idx])

        img = Image.open(img_path).convert("RGB")

        # Find bbox coordinates in bbox_info dataframe
        obj_ids = self.bbox_info.loc[(
            self.bbox_info['img'] == int(self.imgs[idx][:-4]))]

        # get bounding box coordinates for each image
        num_objs = len(obj_ids)
        boxes = []
        for _, bbox in obj_ids.iterrows():
            boxes.append([bbox.x1, bbox.y1, bbox.x2, bbox.y2])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # each image might belong to several classes
        labels = torch.as_tensor(obj_ids.category.tolist(), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_object_detection(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():

    root = "../UECFOOD100merged"
    old_root = "../UECFOOD100"

    # Uncomment the following line to convert dataset's
    # folder structure to the appropriate one. All images should
    # be located at the same directory (ie. UECFOOD100) without
    # any subdirs.

    convert_dataset(old_root, root)

    categ_labels = load_categories(old_root)
    bbox_info = merge_info(old_root)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = len(categ_labels) + 1
    # use our dataset and defined transformations
    dataset = foodDataset(root, get_transform(train=True), bbox_info)
    dataset_test = foodDataset(root, get_transform(train=False), bbox_info)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_object_detection(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # save weights
        torch.save(model.state_dict(), os.path.join('../', 'weights.pt'))
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == "__main__":
    main()

# %%
"""
from matplotlib import pyplot as plt
from PIL import ImageDraw, Image


def get_rect(x1, y1, x2, y2):

    width = abs(x2-x1)
    height = abs(y2-y1)
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = 0
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x1, y1])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect

# pick one image from the test set
img, _ = dataset_test[0]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

imag = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
draw = ImageDraw.Draw(imag)
rect = get_rect(x1=120, y1=80, x2=100, y2=40)
draw.polygon([tuple(p) for p in rect], fill=1)
new_data = np.asarray(imag)

plt.imshow(new_data)
plt.show()
"""

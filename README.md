# Food-Object-Detection-Pytorch-FasterRCNN

This project has been implemented for the purpose of learning and applying the main concepts of object detection (multi-class classification + localization using bounding boxes)

# Instructions - Python 3.6.8 (Windows 10 Pro x64)

<b>!! Important - Numpy 1.17 must be installed (not later version) !!</b>


Clone the Repo:

	git clone https://github.com/kosletr/Food-Object-Detection-Pytorch-FasterRCNN/

Install Requirements by opening a terminal and running:

	pip install -r requirements.txt
  
Download the UECFOOD100 dataset from http://foodcam.mobi/dataset100.html and extract it.

Open terminal and run the script to start the training process:

	python mainObjDet.py


# Example Results

After 10 epochs of training, some of the results produced by the model are shown below:

![](https://github.com/kosletr/testing/blob/master/imgs/Rice.png) 

![](https://github.com/kosletr/testing/blob/master/imgs/Spinach.png)

![](https://github.com/kosletr/testing/blob/master/imgs/Multiple.png)

# Credits
Credits to pytorch.org tutorial found in here:

http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

and the creators of the dataset at:

http://foodcam.mobi/dataset100.html

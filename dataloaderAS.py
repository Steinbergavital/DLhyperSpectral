# -*- coding: utf-8 -*-

from __future__ import print_function, division

# Ignore warnings
import warnings

import h5py as h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


# This is a helper function that displays an image and pauses before attempting to display another one.
def showImgVPause(image):
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated


# torch.data.Dataset is an abstract class representing a dataset.
# Our custom dataset should inherit Dataset. It will have the folloing methods:
# __len__ - len(dataset) returns the size of the dataset.
# __getitem__ - supports the indexing such that dataset[i] can be used to get ith sample
# Here we create a dataset class for our RGBdataset.
# We will read the filenames in __init__
# We will leave the reading of images to __getitem__. (for memory efficiency)

class CHDataset(Dataset):
    """This class returns a dictionary with each RGB image and it corresponding 31-channel hyperspectral dataset."""

    def __init__(self, rgb_img_names, hs_img_names, transform=None):
        """
        Initiates the CHDataset instance.
        Args:
            img_names: The array with the RGB image names in the directory
            hs_img_names: The array with the HS image names in the corresponding HS-directory, make sure corresponding images are in the same order in the RGB and HS directories
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.rgb_img_names = rgb_img_names
        self.hs_img_names = hs_img_names

    def __len__(self):
        # Returns the length of the dataset
        rgb_img_names = self.rgb_img_names
        return len(rgb_img_names)

    def __getitem__(self, idx):
        # This function gets the item from the dictionary, which corresponds to the index given as an argument.
        rgb_img_names = self.rgb_img_names
        hs_img_names = self.hs_img_names
        rgb_img_name = rgb_img_names[idx]
        hs_img_name = hs_img_names[idx]
        rgb_image = io.imread(rgb_img_name)
        rad = get_rad_from_matfile(hs_img_name)
        radRshp = reshape_hs(rad)
        hs_image = radRshp
        sample = {'rgb_image': rgb_image, 'hs_image': hs_image}

        if self.transform:
            sample = self.transform(sample)

        return sample


def getNames(imgDir, imgExt):
    # This function gets the name of a directory (a relative name vs. our home directory) and iterates over
    # the .png files. (or any other extension specified in imgExt) It gets their names and returns a list of .png files.
    import os
    print(imgDir)
    fullnames = []
    for dirName, subdirList, fileList in os.walk(imgDir):
        # for filename in fileList:
        for filename in sorted(fileList):
            nameNoext, ext = os.path.splitext(filename)
            fullfileNm = os.path.join(dirName, filename)
            if (ext == imgExt):
                print(fullfileNm)
                fullnames.append(fullfileNm)
    return fullnames


def get_rad_from_matfile(filename):
    # The .mat file has bands, which is 1*31 and rad, which is 1392*1300*31.
    # We need the rad part, so get_rad_from_matfile returns it as a numpy array.
    matfile = h5py.File(filename, 'r')
    mat = {}
    for k, v in matfile.items():
        # print("key is: " + k + "value is: " + str(v))
        mat[k] = np.array(v)
    return mat['rad']


def reshape_hs(im_hs):
    # Shape the hyperspectral image in the same order as the RGB image.
    image_height = im_hs.shape[2]
    image_width = im_hs.shape[1]
    hs_depth = im_hs.shape[0]
    im_hs = im_hs.transpose(2, 1, 0).reshape(image_height, image_width, hs_depth)
    return im_hs


def chooseChannel(img, ch):
    # Subset the image and return the channel of choice. Tested on RGB images.
    chImg = img[:, :, ch]
    return chImg


def display_hs(image8B, imgChStr):
    # Accepts an 8-bit hyperspectral image as input and a title for the image.
    # Displays an RGB-like representation of the hyperspectral image - it subsets it into 3 channels.
    ax1.set_title(imgChStr)
    plt.imshow(image8B[:, :, [30, 15, 1]].astype('B'), interpolation='nearest')
    plt.show()


def display_rgb(RGBimg, imgChStr):
    ax2.set_title(imgChStr)
    plt.imshow(RGBimg)
    plt.show()


def img12_to8bit(image12B):
    # Accepts a 12-bit (hyperspectral) image, converts it to an 8-bit image and returns it.
    factor = 255 / 4095
    image8B = image12B * factor
    return image8B


def getRGBImgObj(imgName):
    # Get the RGB image name as an argument and return the image object.
    RGBimg = io.imread(imgName)
    return RGBimg


def getRGB_CHObj(imgName, CH):
    # Get the RGB image name and channel number as an argument and return the corresponding channel image object.
    RGBimg = io.imread(imgName)
    RGB_CH = RGBimg[:, :, CH]
    return RGB_CH


def getHSImgObj(imgName):
    rad = get_rad_from_matfile(imgName)
    radRshp = reshape_hs(rad)
    radRshp8B = img12_to8bit(radRshp)
    return radRshp8B


def getHS_CHObj(imgName, CH):
    rad = get_rad_from_matfile(imgName)
    radRshp = reshape_hs(rad)
    radRshp_CH = chooseChannel(radRshp, CH)
    return radRshp_CH


# This will only work if corresponding RGB and HS images appear in the same order
myRGBDir = 'Project_with_Shiran/PyTorch_tutorials/dataBoazRGB_HS/NTIRE2018_Train1_Clean'  # RGB
myHSDir = 'Project_with_Shiran/PyTorch_tutorials/dataBoazRGB_HS/NTIRE2018_Train1_Spectral'  # HS
rgb_img_names = getNames(myRGBDir, '.png')
hs_img_names = getNames(myHSDir, '.mat')
print("There are: " + str(len(rgb_img_names)) + " RGB files.")
print("There are: " + str(len(hs_img_names)) + " .mat files.")
for rgb_img_name in rgb_img_names:
    print("RGB image name: " + rgb_img_name)

for hs_img_name in hs_img_names:
    print("hs image name: " + hs_img_name)

ax1 = plt.subplot(1, 2, 1)
radRshp8B = getHSImgObj(hs_img_names[4])
imgChStr = 'hs_' + 'ch' + str(0)
display_hs(radRshp8B, imgChStr)

ax2 = plt.subplot(1, 2, 2)
RGBimgCH = getRGBImgObj(rgb_img_names[4])
imgChStr = 'rgb_' + 'ch' + str(0)
display_rgb(RGBimgCH, imgChStr)

# Here we instantiate the class and iterate through the data samples.
# We will display the first 3 RGB samples.

rgb_dataset = CHDataset(rgb_img_names=rgb_img_names, hs_img_names=hs_img_names)
fig = plt.figure(figsize=(5, 5))

count = 0
for rowNum in range(3):
    for colNum in range(3):
        count = count + 1
        rowNumPlus1 = rowNum + 1
        colNumPlus1 = colNum + 1
        if colNum == 0:
            CH = 1
        elif colNum == 1:
            CH = 15
        else:
            CH = 30
        image = getHS_CHObj(hs_img_names[rowNum], CH)
        ax = plt.subplot(3, 3, count)
        imgChStr = 'HSsample' + str(rowNumPlus1) + 'ch' + str(colNumPlus1)
        ax.set_title(imgChStr)
        plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=1.0)
        plt.imshow(image)
plt.show()

fig2 = plt.figure(figsize=(5, 5))
count = 0
rowNum = 0
colNum = 0
for rowNum in range(3):
    for colNum in range(3):
        count = count + 1
        rowNumPlus1 = rowNum + 1
        colNumPlus1 = colNum + 1
        RGBimage = io.imread(rgb_img_names[rowNum])
        image = RGBimage[:, :, colNum]
        imgChStr = 'RGBsample' + str(rowNumPlus1) + 'ch' + str(colNumPlus1)
        ax = fig2.add_subplot(3, 3, count)
        ax.set_title(imgChStr)
        plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=1.0)
        plt.imshow(image)
plt.show()


# Transforms - we will create three transforms:
# Rescale: to scale the image
# RandomCrop: to crop from image randomly. This is data augmentation.
# ToTensor: to convert the numpy images to torch images (we need to swap axes).

# We will write the transforms as callable classes instead of simple functions.
# So the parameters of the transform need not be passed everytime it's called:
# tsfm = Transform(params)
# transformed_sample = tsfm(sample)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        rgb_image = sample['rgb_image']
        hs_image = sample['hs_image']

        h, w = rgb_image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        rgb_img = transform.resize(rgb_image, (new_h, new_w))
        hs_img = transform.resize(hs_image, (new_h, new_w))

        return {'rgb_image': rgb_img, 'hs_image': hs_img}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        rgb_image = sample['rgb_image']
        hs_image = sample['hs_image']
        # Extra:

        hRGB, wRGB = rgb_image.shape[:2]
        new_h, new_w = self.output_size

        topRGB = np.random.randint(0, hRGB - new_h)
        leftRGB = np.random.randint(0, wRGB - new_w)

        hHS, wHS = hs_image.shape[:2]
        new_h, new_w = self.output_size

        topHS = np.random.randint(0, hHS - new_h)
        leftHS = np.random.randint(0, wHS - new_w)

        rgb_image = rgb_image[topRGB: topRGB + new_h,
                    leftRGB: leftRGB + new_w]

        hs_image = hs_image[topHS: topHS + new_h,
                   leftHS: leftHS + new_w]

        return {'rgb_image': rgb_image, 'hs_image': hs_image}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb_image = sample['rgb_image']
        hs_image = sample['hs_image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        rgb_image = rgb_image.transpose((2, 0, 1))
        hs_image = hs_image.transpose((2, 0, 1))
        return {'rgb_image': torch.from_numpy(rgb_image),
                'hs_image': torch.from_numpy(hs_image)}


# Now we will apply the transforms on a sample:

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = rgb_dataset[2]
count = 0
for i, tsfrm in enumerate([scale, crop, composed]):
    print("rgb image: " + str(sample['rgb_image'].dtype))
    print("hs image: " + str(sample['hs_image'].dtype))
    transformed_sample = tsfrm(sample)

    count = count + 1
    ax = plt.subplot(2, 3, count)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    trnsImg = transformed_sample['rgb_image']

    trnsImgRed = trnsImg[:, :, 0]
    # trnsImgGreen = trnsImg[:,:,1]
    # trnsImgBlue = trnsImg[:,:,0]
    plt.imshow(trnsImgRed)

    count = count + 1
    ax = plt.subplot(2, 3, count)
    plt.tight_layout()
    hstitle = type(tsfrm).__name__ + "_hs"
    ax.set_title(hstitle)
    trnsImg = transformed_sample['hs_image']
    trnsImgCH1 = trnsImg[:, :, 0]
    plt.imshow(trnsImgCH1)

plt.show()

# Apply each of the above transforms on sample.

#transformed_dataset = CHDataset(rgb_img_names=rgb_img_names,
                               # hs_img_names=hs_img_names,
                               # transform=transforms.Compose([
                              #      Rescale(256),
                              #      RandomCrop(224),
                              #      ToTensor()
                              #  ]))

transformed_dataset = CHDataset(rgb_img_names=rgb_img_names,
                                hs_img_names=hs_img_names,
                                transform=transforms.Compose([
                                    Rescale(256),
                                    ToTensor()
                                ]))


for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['rgb_image'].size())

    if i == 3:
        break

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=False, num_workers=4)

print("After dataloader")

def make_gridHS(HS_batch, nrow=8,padding=2,pad_value=0):
    nmaps = HS_batch.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(HS_batch.size(2) + padding), int(HS_batch.size(3) + padding)
    grid = HS_batch.new(31, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(HS_batch[k])
            k = k + 1
    return grid


# Helper function to show a batch
def show_batch(sample_batch,boolRGB,CH):
    """Show image for a batch of samples."""
    rgb_batch = sample_batch['rgb_image']
    hs_batch = sample_batch['hs_image']
    batch_size = len(rgb_batch)
    im_sizeRGB = rgb_batch.size(2)
    im_sizeHS = hs_batch.size(2)
    if boolRGB == True:
        print("The RGB image shape is: " + str(sample_batch['rgb_image'].shape))

        grid = utils.make_grid(rgb_batch)
        print("grids shape is: " + str(grid.shape))
    else:
        print("image before make_grids shape is: " + str(hs_batch.shape))
        grid = make_gridHS(hs_batch)
        print("grids shape is: " + str(grid.shape))

    trnpGrid = grid.numpy().transpose((1, 2, 0))
    print("trnpGrid shape is: " + str(trnpGrid.shape))

    # Separating the images into channels:
    trnpGridCH = trnpGrid[:, :, CH]
    plt.imshow(trnpGridCH)

    for i in range(batch_size):
        plt.scatter(0 + i * im_sizeRGB, 0, s=10, marker='.', c='r')  # I just wanted it to show
        # the images side by side. There's a red circle at 0,0 for each image instead of a landmark.

        plt.title('Batch from dataloader')


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['rgb_image'].size())

    # observe 1st batch.
    if i_batch == 0:
        plt.figure()
        show_batch(sample_batched, False, 3)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break



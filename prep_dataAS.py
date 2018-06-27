import numpy as np
from os import listdir
import imageio
from scipy import misc
from matplotlib import pyplot as plt
import h5py as h5py
from tqdm import tqdm


def get_ref_from_file(filename):
    matfile = h5py.File(filename, 'r')
    mat = {}
    for k, v in matfile.items():
        mat[k] = np.array(v)
    return mat['rad']


def display_hs(image):
    factor = 255/4095
    image = image*factor
    plt.imshow(image[:, :, [30, 15, 1]].astype('B'), interpolation='nearest')
    plt.show()


# Create two lists of rgb files names and hyperspectral files names
rgb_names = [f for f in listdir('/Users/avitalsteinberg/DeepLearningIsrael/Project_with_Shiran/PyTorch_tutorials/dataBoazRGB_HS/NTIRE2018_Train1_Clean') if f.endswith('.png')]
hs_names = list()
for name in rgb_names:
    print("RGB_name: " + name)
    hs_name = name[:12] + '.mat'
    hs_names.append(hs_name)

num_images = len(rgb_names)


image_width = 1300
image_height = 1392
rgb_depth = 3
hs_depth = 31

# Load data into numpy arrays - input rgb, output hyperspectral

X = np.zeros((num_images, image_height, image_width, rgb_depth))  # rgb images
y = np.zeros((num_images, image_height, image_width, hs_depth))  # hs images

# open rgb and hs images and assign them into numpy array
for idx in tqdm(range(num_images)):
    im_rgb = imageio.imread('/Users/avitalsteinberg/DeepLearningIsrael/Project_with_Shiran/PyTorch_tutorials/dataBoazRGB_HS/NTIRE2018_Train1_Clean/' + rgb_names[idx])  # rgb image
    im_hs = get_ref_from_file('/Users/avitalsteinberg/DeepLearningIsrael/Project_with_Shiran/PyTorch_tutorials/dataBoazRGB_HS/NTIRE2018_Train1_Spectral/' + hs_names[idx])  # hyper-spectral image
    im_hs = im_hs.transpose(2, 1, 0).reshape(image_height, image_width, hs_depth)  # reshape im_hs: from C,H,W to H,W,C

    if im_rgb.shape[1] != image_width:
        im_rgb = misc.imresize(im_rgb, (image_height, image_width), interp='nearest')
        im_hs = misc.imresize(im_hs, (hs_depth, image_height, image_width), interp='nearest')

    X[idx, :, :, :] = im_rgb
    y[idx, :, :, :] = im_hs

print("X shape: " + str(X.shape))
print("y shape: " + str(y.shape))

# save data to file
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('X', data=X)
h5f.create_dataset('y', data=y)
h5f.close()

# imread output is int and X is float, therefore when display the image with imshow
# we need to do casting to unsigned byte ('B')
#plt.imshow(X[3, :, :, :].astype('B'), interpolation='nearest')
#plt.show()
display_hs(y[3, :, :, :])


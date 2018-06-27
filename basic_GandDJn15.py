import h5py
import numpy as np
from scipy import misc
from skimage.transform import resize
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

dtype = torch.FloatTensor  # the CPU datatype
print_every = 100  # Constant to control how frequently we print train loss



def display_hs_ch(image, ch, ch_is_last):
    factor = 255/4095
    image = image*factor
    if ch_is_last:
        image_ch = image[:, :, ch]
    else:
        image_ch = image[ch, :, :]
    plt.imshow(image_ch, cmap='gray')
    plt.show()
    return image_ch

def h5_to_data(file_name):

    f = h5py.File(file_name, 'r')

    # Names of the groups in HDF5 file
    for key in f.keys():
        print(key)

    X = f['X'][()]
    y = f['y'][()]

    print(X.shape)
    print(y.shape)
    print(type(X))
    print(type(y))

    display_hs_ch(y[0], 18, True)

    X_new = np.zeros((5, 100, 100, 3))  # rgb images
    y_new = np.zeros((5, 100, 100, 31))  # hs images

    for idx in range(5):
        X_new[idx] = resize(X[idx], (100, 100, 3))
        y_new[idx] = resize(y[idx], (100, 100, 31))

    #display_hs_ch(y_new[0], 18, True)

    print(X.shape)
    print(y.shape)

    return X_new, y_new

filename = 'data.h5'
X, y = h5_to_data(filename)
display_hs_ch(y[0], 18, True) # for debugging

X = X.transpose(0, 3, 1, 2).reshape(5, 3, 100, 100)  # reshape X: from N,H,W,C to N,C,H,W
y = y.transpose(0, 3, 1, 2).reshape(5, 31, 100, 100)  # reshape y: from N,H,W,C to N,C,H,W

# display_hs_ch(y[0], 18, False)  # for debugging
# split data to train and test
X_train = torch.from_numpy(X[:3]).type(dtype)
X_test = torch.from_numpy(X[3:]).type(dtype)
y_train = torch.from_numpy(y[:3]).type(dtype)
y_test = torch.from_numpy(y[3:]).type(dtype)

print(('X_train: ', X_train.shape))
print(('y_train: ', y_train.shape))
print(('X_test: ', X_test.shape))
print(('y_test: ', y_test.shape))

# Hyper Parameters
epoch = 10
num_iter = 1000
batch_size = 1
learning_rate = 1e-3


# Initializing the weights for a network that is of the PyTorch nn.Module type:
# Documentation: https://pytorch.org/docs/stable/nn.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class G(nn.Module): # Shiran called it autoencoder - I renamed it
    def __init__(self):
        super(G, self).__init__()
        # encoder input = C - 3, H - 100, W - 100
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),  # output size = C - 64, H - 100, W - 100
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=1),  # output size = C - 64, H - 90, W - 90
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 31, 3, stride=1),  # output size = C - 31, H - 100 , W -100
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#Defining the descriminator class, which is a PyTorch neural network class torch.nn.Module, as described in
# the docs https://pytorch.org/docs/stable/nn.html . The forward class will return either ones or zeros, 
# since I used a Sigmoid function. 
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
        nn.Conv2d(31, 64, 3, 1, 1),
        nn.ReLU(0.2, inplace = True),
        nn.BatchNorm2d(100),
        nn.Conv2d(31, 64, 3, 1, 1),
        nn.Sigmoid()
    )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)



netG = G().type(dtype)   # model = G().cuda(), and I renamed model, it's now called netG
# Each torch.nn.Module class has an apply function, and it is mainly used to initialize the weights
# as explained in the documentation about apply(fn): https://pytorch.org/docs/stable/nn.html
#  netG.apply(weights_init)
criterion = nn.MSELoss().type(dtype)
optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, weight_decay=1e-5) # I renamed optimizer, it's now called optimizerG

# An instance netD of the discriminator class is defined.
# The weights are initialized. The criterion for the loss is the mean square error loss. The optimizer is Adam.:

#netD = D().type(dtype)   # model = D().cuda()
#netD.apply(weights_init)
#criterion = nn.MSELoss().type(dtype)
#optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, weight_decay=1e-5)
'''
loss_history = []

# initialize figure
f, a = plt.subplots(2, 2, figsize=(2, 2))
plt.ion()   # continuously plot

for iteration in range(num_iter):
    img_input = Variable(X_train)
    realOut = Variable(y_train) # I renamed img_output as real_output
    # ===================forward=====================
    # The gradients for the generator class are initialized. 
    netG.zero_grad()
    genOut = netG(img_input) # genOut used to be called output
    errG_REG = criterion(genOut, realOut) # errG_REG used to be called loss
    # ===================backward====================
    
    # We get a real image of the dataset which will be used to train the discriminator, and then wrap it in a variable.
    #Then we forward propagate this real image into the neural network of the discriminator to get the prediction
    #(a value between 0 and 1) and compute the loss between the predictions (output) and the target (equal to 1).
    #netD.zero_grad()
    #img_input = Variable(X_train)
    #target = Variable(torch.ones(img_input.size()[0]))
    #output = netD(img_input)
    #errD_real = criterion(output, target)
    
    # Here we forward propagate the RGB input image into the neural network of the generator to get some fake generated images.
    #Then we forward propagate the fake generated images into the neural network of the discriminator to get the prediction
    #(a value between 0 and 1) and compute the loss between the prediction (output) and the target (equal to 0).
    #fake = netG(img_input)
    #target = Variable(torch.zeros(img_input.size()[0]))
    #output = netD(fake.detach())
    #errD_fake = criterion(output, target)
    
    # Back-propagating the total error :
    #errD = errD_real + errD_fake
    #errD.backward()
    #optimizerD.step()

    #Getting the target. Forward propagating the fake generated images into the neural network of the discriminator to get
    #the prediction (a value between 0 and 1) and then computing the loss between the prediction (output between 0 and 1) 
    #and the target (equal to 1). 
    #target = Variable(torch.ones(input.size()[0]))
    #output = netD(fake)
    #errG_GAN = criterion(output, target)
    #errG = errG_GAN + errG_REG
    errG = errG_REG # Remove this line when using the GAN error for the generator
    
    #The next step is to update the weights of the neural network of the generator :
    optimizerG.zero_grad()
    errG.backward()
    optimizerG.step()
    # ===================log========================

    if iteration % 10 == 0:
        print('iteration [{}/{}], loss:{:.4f}'
              .format(iteration, num_iter, errG_REG.data[0]))
        loss_history.append(errG_REG.data[0])

        # ===================Plotting========================
        image_for_plt_real = y_train[0].numpy()  # for plotting - real hs image
        image_for_plt_gen = genOut[0].data.numpy()  # for plotting - generated hs image

        a[0][0].clear()
        a[1][0].clear()
        a[1][1].clear()
        a[0][0].imshow(display_hs_ch(image_for_plt_real, 18, False), cmap='gray')
        a[0][0].set_xticks(())
        a[0][0].set_yticks(())
        a[1][0].imshow(display_hs_ch(image_for_plt_gen, 18, False), cmap='gray')
        a[1][0].set_xticks(())
        a[1][0].set_yticks(())
        a[1][1].plot(loss_history)
        plt.draw()
        plt.pause(0.05)

        # plt.plot(loss_history)
        # plt.savefig('loss.jpg')
        # plt.close()

    #        pic = to_img(output.cpu().data)
    #        save_image(pic, './dc_img/image_{}.png'.format(epoch))

print('Done iterating')
plt.ioff()
plt.show()
torch.save(netG.state_dict(), './conv_autoencoder.pth')

    # https://github.com/SherlockLiao/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
    
    # GAN tutorial: https://becominghuman.ai/understanding-and-building-generative-adversarial-networks-gans-8de7c1dc0e25
'''
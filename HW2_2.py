#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from random import randint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:0')

#%%
batch_size = 64
latent_code = 32   

data_path = '../anime-faces/data/'
transform = transforms.Compose(
    [
        transforms.Resize((64,64)), # resize images to same size
        torchvision.transforms.ToTensor(), # image to tensor
    ]
)
#%%
# dataset
dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=transform
)

# train test split
num_data = len(dataset)
ratio = .9
trainSet, testSet = torch.utils.data.random_split(dataset, [int(num_data*ratio), num_data-int(num_data*ratio)])
print("number of train data:", len(trainSet))
print("number of test data:", len(testSet))

# dataloader
trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
)
testLoader = torch.utils.data.DataLoader(
    testSet,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
)

#%%
class flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class unflatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

hd = 1024
zd = 32 # latent

dims = [3, 32, 64, 128, 256] # dims[0]為image的channel數

STRIDE = 2

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], 4, STRIDE),
            nn.BatchNorm2d(dims[1]),
            nn.ReLU(),
            nn.Conv2d(dims[1], dims[2], 4, STRIDE),
            nn.BatchNorm2d(dims[2]),
            nn.ReLU(),
            nn.Conv2d(dims[2], dims[3], 4, STRIDE),
            nn.BatchNorm2d(dims[3]),
            nn.ReLU(),
            nn.Conv2d(dims[3], dims[4], 4, STRIDE),
            nn.BatchNorm2d(dims[4]),
            nn.ReLU(),
            flatten()
        )
        self.decoder = nn.Sequential(
            unflatten(),
            nn.ConvTranspose2d(hd, dims[3], 5, STRIDE),
            nn.ReLU(),
            nn.ConvTranspose2d(dims[3],  dims[2], 5, STRIDE),
            nn.ReLU(),
            nn.ConvTranspose2d(dims[2], dims[1], 6, STRIDE),
            nn.ReLU(),
            nn.ConvTranspose2d(dims[1], dims[0], 6, STRIDE),
            nn.Sigmoid(),
        )
        self.fc_1 = nn.Linear(hd, zd)
        self.fc_2 = nn.Linear(hd, zd)
        self.fc_3 = nn.Linear(zd, hd)

#         self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        z, mean, var = self.encode(x)
        z = self.decode(z)
        return z, mean, var

    def encode(self, x):
        h = self.encoder(x)
        z, mean, var = self.get_latent_z(h)
        return z, mean, var

    def decode(self, z):
        z = self.fc_3(z)
        z = self.decoder(z)
        return z    
    
    def get_latent_z(self, h):
        mean, var = self.fc_1(h), self.fc_2(h)
        std = var.mul(0.5).exp_()
        esp = torch.cuda.FloatTensor(std.size()).normal_() 
        z = mean + std * esp
        return z, mean, var


# %%
epochs = 100
learning_rate = 1e-3
save_interval = 100
loss_record = []
bce_record = []
kld_record = []

# %%
# 建立資料夾
if not os.path.exists('vae_results'):
    os.makedirs('vae_results')

# 設定實驗kl_term 建立對應資料夾
kl_term = 500
exp_name = 'kl_'+str(kl_term) # kl _1, kl_100, kl_0
if not os.path.exists('vae_results/'+str(exp_name)):
    os.makedirs('vae_results/'+str(exp_name))
if not os.path.exists('vae_results/'+str(exp_name)+'/train'):
    os.makedirs('vae_results/'+str(exp_name)+'/train')
# 建立sample z的資料夾
if not os.path.exists('vae_results/'+str(exp_name)+'/sample_z'):
    os.makedirs('vae_results/'+str(exp_name)+'/sample_z')

# %%
import os
def interpolation(x1,x2):
    if not os.path.exists('vae_results/'+str(exp_name)+'/interpolation'):
        os.makedirs('vae_results/'+str(exp_name)+'/interpolation')
    size = 8
    x1 = x1.view(1,3,64,64)
    h1 = vae.encoder(x1)
    z1, _, _ = vae.get_latent_z(h1)
    
    x2 = x2.view(1,3,64,64)
    h2 = vae.encoder(x2)
    z2, _, _ = vae.get_latent_z(h2)
    
    interpolate = z1
    
    for  i in range(1,size): # 8*32
        interpolate = torch.cat([interpolate,z1+z2*(i/size)+z1*((size-i)/size)], 0)
    
    origin_img = torch.cat([x1,x2])
    save_image(origin_img.data.cpu().view(2, 3, 64, 64), './vae_results/'+str(exp_name)+'/interpolation/epoch_' +str(epoch) +'_'+str(idx)+ '_origin.png')
    
    # 10 z to decoder
    z_recon = vae.decoder(vae.fc_3(interpolate).view(size, 1024, 1, 1))
    save_image(z_recon.data.cpu().view(size, 3, 64, 64), './vae_results/'+str(exp_name)+'/interpolation/epoch_' +str(epoch) +'_'+str(idx)+ '_z_const.png')
    print("[interpolation] Save samples finished......")

# %%
def cal_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, (x),  size_average=False) #,  size_average=False .sigmoid
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD*kl_term, BCE, KLD

# %%
vae = VAE().to(device)# vae.cuda()

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# sample z
sample = Variable(torch.randn(batch_size, 32))

for epoch in range(epochs):
    train_loss = 0
    test_loss = 0
    bce_loss = 0
    kld_loss = 0

    # for training...
    for idx, (imgs, _) in enumerate(trainLoader):
        imgs = imgs.to(device)
        recon_imgs, mu, logvar = vae(imgs)
        loss, bce, kld = cal_loss(recon_imgs, imgs, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        bce_loss += bce.item()
        kld_loss += kld.item()

        if idx % save_interval == 0:
            # save interpolation images
            interpolation(imgs[0],imgs[1])
            # save train images
            save_image(imgs.data.cpu().view(imgs.shape[0], 3, 64, 64), './vae_results/'+str(exp_name)+'/train/epoch_' +str(epoch) + '_'+str(idx)+'_origin.png')
            save_image(recon_imgs.data.cpu().view(imgs.shape[0], 3, 64, 64),  './vae_results/'+str(exp_name)+'/train/epoch_' +str(epoch) + '_recon.png')
            print("[train images] Save samples finished......")
            
            to_print = "Epoch[{}/{}] Loss: {:.3f} BCE: {:.3f} KL: {:.3f}".format(epoch+1, epochs, loss.item()/batch_size, bce.item()/batch_size, kld.item()/batch_size) #/batch_size
            print(to_print)
            
    # sample z
    sample = sample.cuda()
    sample_recon = vae.decoder(vae.fc_3(sample).view(batch_size, 1024, 1, 1))
    save_image(sample_recon.data.cpu().view(batch_size, 3, 64, 64), './vae_results/'+str(exp_name)+'/sample_z/epoch_' +str(epoch) + '.png')
    print("[sample z] Save samples finished......")
    
    # save record
    loss_record.append(train_loss/len(trainSet))
    bce_record.append(bce_loss/len(trainSet))
    kld_record.append(kld_loss/len(trainSet))
    
    to_print = "Epoch[{}/{}] Training Loss: {:.3f}".format(epoch+1,epochs, train_loss/len(trainSet))
    print(to_print)
    
    # for testing... 
    for idx, (imgs, _) in enumerate(testLoader):
        imgs = imgs.cuda()
        recon_imgs, mu, logvar = vae(imgs)
        loss, _, _ = cal_loss(recon_imgs, imgs, mu, logvar)
        test_loss += loss
    
    to_print = "Epoch[{}/{}] Testing Loss: {:.3f}".format(epoch+1,epochs, test_loss/len(testSet))
    print(to_print)

    print('-----------------------------------------------------')


# %%
'''save result'''
import pickle
#  save record
plt.plot(loss_record)
plt.title('loss record')
plt.xlabel('epochs')
# loss_record.to_pickle('./vae_results/'+str(exp_name)+'/loss.pkl')
with open('./vae_results/'+str(exp_name)+'/loss.pkl', 'wb') as b:
    pickle.dump(loss_record,b)
plt.savefig('./vae_results/'+str(exp_name)+'/loss_record.png')
plt.show()

plt.plot(bce_record)
plt.title('bce loss')
plt.xlabel('epochs')
# bce_record.to_pickle('./vae_results/'+str(exp_name)+'/bce.pkl')
with open('./vae_results/'+str(exp_name)+'/bce.pkl', 'wb') as b:
    pickle.dump(bce_record,b)
plt.savefig('./vae_results/'+str(exp_name)+'/bce_record.png')
plt.show()

plt.plot(kld_record)
plt.title('kld loss')
plt.xlabel('epochs')
# kld_record.to_pickle('./vae_results/'+str(exp_name)+'/kld.pkl')
with open('./vae_results/'+str(exp_name)+'/kld.pkl', 'wb') as b:
    pickle.dump(kld_record,b)
plt.savefig('./vae_results/'+str(exp_name)+'/kld_record.png')
plt.show()

# %%
import pandas as pd
kl_0 = pd.read_pickle('./vae_results/'+str('kl_0')+'/loss.pkl')
kl_1 = pd.read_pickle('./vae_results/'+str('kl_1')+'/loss.pkl')
kl_100 = pd.read_pickle('./vae_results/'+str('kl_100')+'/loss.pkl')
kl_500 = pd.read_pickle('./vae_results/'+str('kl_500')+'/loss.pkl')

kl_0 = pd.read_pickle('./vae_results/'+str('kl_0')+'/bce.pkl')
kl_1 = pd.read_pickle('./vae_results/'+str('kl_1')+'/bce.pkl')
kl_100 = pd.read_pickle('./vae_results/'+str('kl_100')+'/bce.pkl')
kl_500 = pd.read_pickle('./vae_results/'+str('kl_500')+'/bce.pkl')

kl_0 = pd.read_pickle('./vae_results/'+str('kl_0')+'/kld.pkl')
kl_1 = pd.read_pickle('./vae_results/'+str('kl_1')+'/kld.pkl')
kl_100 = pd.read_pickle('./vae_results/'+str('kl_100')+'/kld.pkl')
kl_500 = pd.read_pickle('./vae_results/'+str('kl_500')+'/kld.pkl')

plt.plot(kl_0, label='kl_0')
plt.plot(kl_1, label='kl_1')
plt.plot(kl_100, label='kl_100')
plt.plot(kl_500, label='kl_500')
plt.legend(loc='upper right')
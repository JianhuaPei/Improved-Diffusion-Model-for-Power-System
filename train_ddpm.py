import os
import glob
import math
from abc import abstractmethod


import requests
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


from sklearn import preprocessing
from model import UNetModel, GaussianDiffusion
from Config import dic_obj as opt

#load dataset
class Dataset_m(Dataset):
    def __init__(self, root):
        self.root = root
        self.data_list = glob.glob(f'{self.root}/*.csv')
        pass

    def __getitem__(self, index):
        data = self.data_list[index]
        data = pd.read_csv(data)
        data = data.values
        data = data[:,0:40]
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1), copy=False)
        data = min_max_scaler.fit_transform(data)
        data = data.T
        data = data.reshape(1, opt.inputdata_size_M, opt.inputdata_size_T)
        data = np.array(data, dtype = np.float32)        
        data = torch.tensor(data)
        return data

    def __len__(self):
        return len(self.data_list)


if __name__=='__main__':
    

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5]) ])
    
    # use MNIST dataset
    # dataset = datasets.MNIST('F://data', train=True, download=True, transform=transform)
    
    my_dataset = Dataset_m(f'{opt.data_path}')
    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size = opt.batch_size_ddpm, shuffle = True)


    #define model and diffusion
    device = "cpu"
    model = UNetModel(in_channels = 1, model_channels = 128, out_channels = 1, channel_mult = (1,2,2), attention_resolutions = [] )
    model.to(device)
    
    
    gaussian_diffusion = GaussianDiffusion(timesteps = opt.timesteps_ddpm)
    optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr_ddpm)
    
    #train 
    for epoch in range(opt.epochs_ddpm):
        for step, measurements in enumerate(train_loader):
            optimizer.zero_grad()
            batch_size = measurements.shape[0]
            measurements = measurements.to(device)
            iteration = epoch*len(train_loader) + step
            
        
            #sample n uniformally for every example in the batch
            n = torch.randint(0, opt.timesteps_ddpm, (batch_size,), device = device).long()
        
            loss = gaussian_diffusion.train_losses(model, measurements, n)
            if iteration % 100 == 0:
                    print (
                        "[Epoch %d/%d] [iteration %d] [Loss: %f] "
                        %(epoch, opt.epochs_ddpm, iteration, loss.item())
                        )

            
            loss.backward()
            optimizer.step()
    
    
    #save DDPM model
    torch.save(model.state_dict(), 'g:\\Python\\Diffusion_Models\\improved_diffusion_model_power_system\\saved_model\\ddpm_model_of_case_39_loadchanging.pth')
    
    
    
    
import os
import math
from abc import abstractmethod
import glob
import random

from PIL import Image
import requests
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing

from model import UNetModel, GaussianDiffusion
from Config import dic_obj as opt

def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


if __name__=='__main__':
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    #load the trained model
    model = UNetModel(in_channels=1, model_channels=96, out_channels=1, channel_mult=(1,2,2), attention_resolutions=[] )
    device = 'cpu'
    model.to(device)
    model.load_state_dict(torch.load(f'g:\\Python\\Diffusion_Models\\improved_diffusion_model_power_system\\saved_model\\ddpm_model_of_case_140.pth'))
    model.eval()
    
    
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])   ])
    #dataset = datasets.MNIST('F://data', train=True, download=True, transform=transform)
    gaussian_diffusion = GaussianDiffusion(timesteps = opt.timesteps_ddpm)
    
    len_dataset = 2000
    ms_eps = np.zeros( opt.timesteps_ddpm, dtype=np.float32)
    for n in range(0, opt.timesteps_ddpm ):
        eps_sum = 0
        for i in range(1, len_dataset+1):
            data_path = glob.glob(f'{opt.data_path}/**/*.csv')
            random_number = random.randint(0, 16000)
            x_start = pd.read_csv(data_path[random_number])
            x_start = x_start.values
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1), copy=False)
            x_start = min_max_scaler.fit_transform(x_start)
            x_start = x_start.T
            x_start = np.array(x_start, dtype = np.float32)  
            x_start = torch.tensor([x_start], device = device)
            n_tensor = torch.tensor([n], device = device)
            x_n = gaussian_diffusion.q_sample(x_start, n_tensor)
            x_n = x_n.to(device)
            x_n = torch.reshape(x_n, [1,1,opt.inputdata_size_M,opt.inputdata_size_T])
            pred_noise = model(x_n, n_tensor)
            pred_noise = pred_noise.detach().numpy()
            eps_sum = eps_sum + np.sum(pred_noise**2)/(opt.inputdata_size_M*opt.inputdata_size_T)
            
            print (
                        "[n %d/%d] [iteration %d/%d] "
                        %(n, opt.timesteps_ddpm - 1, i, len_dataset)
                        )
            
        ms_eps[n] = eps_sum / len_dataset
        print(ms_eps)
    

    np.save("g:\\Python\\Diffusion_Models\\improved_diffusion_model_power_system\\saved_data\\ms_eps_of_case_140.npy",ms_eps)
    
        
            

                                              
            
        
        
        
    
    
import os
import math
from abc import abstractmethod


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




if __name__=='__main__':
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    #load the trained model
    model = UNetModel(in_channels=1, model_channels=96, out_channels=1, channel_mult=(1,2,2), attention_resolutions=[] )
    model.load_state_dict(torch.load(f'g:\\Python\\Diffusion_Models\\improved_diffusion_model_power_system\\saved_model\\ddpm_model_of_case_30.pth'))
    model.eval()
    device = 'cpu'
    
    
    # ddpm case
    gaussian_diffusion = GaussianDiffusion(timesteps = opt.timesteps_ddpm)
    
    
    '''
    generated_images = gaussian_diffusion.sample(model, 28, batch_size=16, channels =1)
    # generate new images
    fig = plt.figure(figsize =(12,12), constrained_layout = True)
    gs = fig.add_gridspec(4,4)

    imgs = generated_images[-1].reshape(4,4,28,28)
    for n_row in range(4):
        for n_col in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((imgs[n_row, n_col]+1.0)*255/2, cmap="gray")
            f_ax.axis("off")
    plt.savefig('F://data//savefig_example_2.png')   


    #show the denoise steps
    fig = plt.figure(figsize=(12,12), constrained_layout=True)     
    gs = fig.add_gridspec(16,16)

    for n_row in range(16):
        for n_col in range(16):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            t_idx = (opt.timesteps_ddpm // 16) *n_col if n_col < 15 else -1
            img = generated_images[t_idx][n_row].reshape(28,28)
            f_ax.imshow((img+1.0)*255/2, cmap="gray")
            f_ax.axis("off")
    plt.savefig('F://data//savefig_example_3.png')      
    
    #naive ddim case
    ddim_generated_images = gaussian_diffusion.ddim_sample(model, 28, batch_size = 16, channels = 1, ddim_timesteps = 50)

    # generate new images
    fig = plt.figure(figsize = (12,12), constrained_layout = True)
    gs = fig.add_gridspec(4,4)

    imgs = ddim_generated_images.reshape(4, 4, 28, 28)
    for n_row in range(4):
        for n_col in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((imgs[n_row, n_col]+1.0)*255 /2, cmap='gray')
            f_ax.axis("off")
    plt.savefig('F://data//savefig_example_4.png')     
    
    
    #ddim with optimal variance 
    #ms_eps = np.load('g:\\Python\\Diffusion_Models\\improved_diffusion_model\\saved_data\\ms_eps.npy')
    
    ddim_generated_images = gaussian_diffusion.ddim_sample_with_optimal_variance(model, 28, batch_size = 16, channels = 1, ddim_timesteps = 50)

    # generate new images
    fig = plt.figure(figsize = (12,12), constrained_layout = True)
    gs = fig.add_gridspec(4,4)

    imgs = ddim_generated_images.reshape(4, 4, 28, 28)
    for n_row in range(4):
        for n_col in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((imgs[n_row, n_col]+1.0)*255 /2, cmap='gray')
            f_ax.axis("off")
    plt.savefig('F://data//savefig_example_5.png')     
    
    '''
    
    #load test data
    data_path_case30 = 'F:\\Renewable_generation_and_load_data\\load_data_process\\Regular_data\\SCADA_data_of_case_30_test\\'
    data_path_case57 = 'F:\\Renewable_generation_and_load_data\\load_data_process\\Regular_data\\SCADA_data_of_case_57_test\\'
    data_path_case118 = 'F:\\Renewable_generation_and_load_data\\load_data_process\\Regular_data\\SCADA_data_of_case_118_test\\'
    data_path_case39 = 'F:\\Renewable_generation_and_load_data\\load_data_process\\Python_based_simulation\\IEEE_39_bus_test_system\\ieee39_test\\'
    data_path_case118 = 'F:\\Renewable_generation_and_load_data\\load_data_process\\Python_based_simulation\\NPCC_140_bus_test_system\\npcc140_test\\'
    
    Slack_30 = 1
    PV_30 = [2,13,22,23,27]
    PV_30_all = [2,13,22,23,27]
    PQ_30 = [5,10,15,28,29]
    PQ_30_all = [3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,24,25,26,28,29,30]
    Line_30 = [10]
    Slack_57 = 1
    PV_57 = [2,3,6,8,9,12]
    PV_57_all = [2,3,6,8,9,12]
    PQ_57 = [4,5,15,19,25,26,29,35,36,39,47,57]
    PQ_57_all = [4,5,7,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]
    Line_57 = [15,29,54,60,79]
    Slack_118 = 69
    PV_118 = [6,10,24,27,31,42,46,55,59,72,76,87,99,107,110]
    PV_118_all = [1,4,6,8,10,12,15,18,19,24,25,26,27,31,32,34,36,40,42,46,49,54,55,56,59,61,62,65,66,70,72,73,74,76,77,80,85,87,89,90,91,92,99,100,103,104,105,107,110,111,112,113,116]
    PQ_118 = [13,21,43,45,53,60,67,68,78,83,84,86,93,94,96,98,101,102,106,108]
    PQ_118_all = [2,3,5,7,9,11,13,14,16,17,20,21,22,23,28,29,30,33,35,37,38,39,41,43,44,45,47,48,50,51,52,53,57,58,60,63,64,67,68,71,75,78,79,81,82,83,84,86,88,93,94,95,96,97,98,101,102,106,108,109,114,115,117,118]
    Line_118 = [17, 21, 43, 62, 67, 70, 72, 91, 140, 164, 176, 185]
    
    index_case30 = []
    index_case30.append(4*Slack_30 -2) #Pi
    index_case30.append(4*Slack_30 -1 ) #Qi
    for i in PV_30:
        index_case30.append(4*i -3) #deltai
        index_case30.append(4*i -1) #Qi
    for i in PQ_30:
        index_case30.append(4*i -4) #Vi
        index_case30.append(4*i -3) #deltai
    for i in Line_30:
        index_case30.append(30*4 + 2*i -2) #Pij
        index_case30.append(30*4 + 2*i -1) #Qij
    
    index_case30_ori = []
    index_case30_ori.append(2) #Pi
    index_case30_ori.append(3) #Qi
    for i in PV_30:
        index_case30_ori.append(4+(PV_30_all.index(i)+1)*4-3)  #deltai
        index_case30_ori.append(4+(PV_30_all.index(i)+1)*4-1)  #Qi
    for i in PQ_30:
        index_case30_ori.append(4+4*len(PV_30_all)+(PQ_30_all.index(i)+1)*4 -4) #Vi
        index_case30_ori.append(4+4*len(PV_30_all)+(PQ_30_all.index(i)+1)*4 -3) #deltai
    for i in Line_30:
        index_case30_ori.append(30*4 + 2*i -2) #Pij
        index_case30_ori.append(30*4 + 2*i -1) #Qij
    
    
    
    index_case57 = []
    index_case57.append(4*Slack_57 -2) #Pi
    index_case57.append(4*Slack_57 -1 ) #Qi
    for i in PV_57:
        index_case57.append(4*i -3) #deltai
        index_case57.append(4*i -1) #Qi
    for i in PQ_57:
        index_case57.append(4*i -4) #Vi
        index_case57.append(4*i -3) #deltai
    for i in Line_57:
        index_case57.append(57*4 + 2*i -2) #Pij
        index_case57.append(57*4 + 2*i -1) #Qij  
        
    index_case57_ori = []
    index_case57_ori.append(2) #Pi
    index_case57_ori.append(3) #Qi
    for i in PV_57:
        index_case57_ori.append(4+(PV_57_all.index(i)+1)*4-3)  #deltai
        index_case57_ori.append(4+(PV_57_all.index(i)+1)*4-1)  #Qi
    for i in PQ_57:
        index_case57_ori.append(4+4*len(PV_57_all)+(PQ_57_all.index(i)+1)*4 -4) #Vi
        index_case57_ori.append(4+4*len(PV_57_all)+(PQ_57_all.index(i)+1)*4 -3) #deltai
    for i in Line_57:
        index_case57_ori.append(57*4 + 2*i -2) #Pij
        index_case57_ori.append(57*4 + 2*i -1) #Qij        
          
    
    index_case118 = []
    index_case118.append(4*Slack_118 -2) #Pi
    index_case118.append(4*Slack_118 -1 ) #Qi
    for i in PV_118:
        index_case118.append(4*i -3) #deltai
        index_case118.append(4*i -1) #Qi
    for i in PQ_118:
        index_case118.append(4*i -4) #Vi
        index_case118.append(4*i -3) #deltai
    for i in Line_118:
        index_case118.append(118*4 + 2*i -2) #Pij
        index_case118.append(118*4 + 2*i -1) #Qij 
        
    index_case118_ori = []
    index_case118_ori.append(2) #Pi
    index_case118_ori.append(3) #Qi
    for i in PV_118:
        index_case118_ori.append(4+(PV_118_all.index(i)+1)*4-3)  #deltai
        index_case118_ori.append(4+(PV_118_all.index(i)+1)*4-1)  #Qi
    for i in PQ_118:
        index_case118_ori.append(4+4*len(PV_118_all)+(PQ_118_all.index(i)+1)*4 -4) #Vi
        index_case118_ori.append(4+4*len(PV_118_all)+(PQ_118_all.index(i)+1)*4 -3) #deltai
    for i in Line_118:
        index_case118_ori.append(118*4 + 2*i -2) #Pij
        index_case118_ori.append(118*4 + 2*i -1) #Qij            
        

    
    
    data_path_test_ori = data_path_case30 + 'original_data\\test_original_data_15.csv'
    data_path_test_step = data_path_case30 + 'modified_num_1\\non_random_loss\\test_case_15.csv'
    
    data_df_ori = pd.read_csv(data_path_test_ori)
    data_np_ori = data_df_ori.values
    
    data_df_step = pd.read_csv(data_path_test_step)
    data_np_step = data_df_step.values
    data_np_input = data_np_step[:, index_case30_ori]
    data_np_input_ori = data_np_ori[:, index_case30_ori]
    
    #conditioned ddim
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1), copy=False)
    data_ts_input = data_np_input.copy()
    
    data_ts_input = min_max_scaler.fit_transform(data_ts_input)
    data_ts_input = data_ts_input.T
    data_ts_input = data_ts_input.reshape(1,1, 24, 96)
    data_ts_input = np.array(data_ts_input, dtype = np.float32)
    data_ts_input = torch.tensor(data_ts_input)
    #ddim_generated = gaussian_diffusion.conditioned_ddim(model, 24, 96, batch_size = 1, channels = 1, ddim_timesteps = 20, y_start = data_ts_input, omega = 2.0)
    ddim_generated = gaussian_diffusion.ddim_imputation(model, 24,96, batch_size = 1, channels = 1, ddim_timesteps = 20, y_start = data_ts_input, R = 5)
    recovered_data = ddim_generated.reshape(24,96)
    recovered_data = recovered_data.T
    recovered_data = min_max_scaler.inverse_transform(recovered_data)

    
    
    x = np.arange(0,96)
    fig1 = plt.figure(num = 'test')
    plt.plot(x, data_np_input[:,2])
    plt.plot(x, data_np_input_ori[:,2])
    plt.plot(x, recovered_data[:,2])
    plt.savefig('F://data//test_1.svg')  
    
    
    
    
    
    
    
    
    
    '''
        
    # conditioned ddim    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])   ])
    dataset = datasets.MNIST('F://data', train=True, download=True, transform=transform)
    y_start = dataset[2]
    y_start = y_start[0]
    y_start = y_start.to(device)
    y_start = torch.reshape(y_start, [1,1,28,28])
    
    y_start_modified = y_start.clone()
    y_start_modified = y_start_modified.numpy()
    y_start_modified[:,:,13:21,13:21] = -1.0
    y_start_modified = torch.tensor(y_start_modified)
    
    ddim_generated_images = gaussian_diffusion.conditioned_ddim(model, 28, batch_size = 1, channels = 1, ddim_timesteps = 50, y_start = y_start_modified, omega = 1.0)


    fig = plt.figure(figsize = (12,12), constrained_layout = True)
    gs = fig.add_gridspec(3,1)    
    
    original_imgs = y_start.numpy()
    modified_imgs = y_start_modified.numpy()
    recovered_imgs = ddim_generated_images.reshape(1, 1, 28, 28)
    
    f_ax = fig.add_subplot(gs[0, 0])
    f_ax.imshow((original_imgs[0, 0]+1.0)*255 /2, cmap='gray')
    f_ax.axis("off")
    
    f_ax = fig.add_subplot(gs[1, 0])
    f_ax.imshow((modified_imgs[0, 0]+1.0)*255 /2, cmap='gray')
    f_ax.axis("off")
    
    f_ax = fig.add_subplot(gs[2, 0])
    f_ax.imshow((recovered_imgs[0, 0]+1.0)*255 /2, cmap='gray')
    f_ax.axis("off")
    
    plt.savefig('F://data//savefig_example_6.png')  
    
    
    # ddim imputation
    y_start = dataset[2]
    y_start = y_start[0]
    y_start = y_start.to(device)
    y_start = torch.reshape(y_start, [1,1,28,28])
    
    y_start_modified = y_start.clone()
    y_start_modified = y_start_modified.numpy()
    y_start_modified[:,:,13:21,13:21] = np.nan
    y_start_modified = torch.tensor(y_start_modified)
    
    ddim_generated_images = gaussian_diffusion.ddim_imputation(model, 28, batch_size = 1, channels = 1, ddim_timesteps = 50, y_start = y_start_modified, R = 5)


    fig = plt.figure(figsize = (12,12), constrained_layout = True)
    gs = fig.add_gridspec(3,1)    
    
    original_imgs = y_start.numpy()
    modified_imgs = y_start_modified.numpy()
    recovered_imgs = ddim_generated_images.reshape(1, 1, 28, 28)
    
    f_ax = fig.add_subplot(gs[0, 0])
    f_ax.imshow((original_imgs[0, 0]+1.0)*255 /2, cmap='gray')
    f_ax.axis("off")
    
    f_ax = fig.add_subplot(gs[1, 0])
    f_ax.imshow((modified_imgs[0, 0]+1.0)*255 /2, cmap='gray')
    f_ax.axis("off")
    
    f_ax = fig.add_subplot(gs[2, 0])
    f_ax.imshow((recovered_imgs[0, 0]+1.0)*255 /2, cmap='gray')
    f_ax.axis("off")
    
    plt.savefig('F://data//savefig_example_7.png')      
    
    '''   
     
    
    
     
    
    
    

     

    
    
    
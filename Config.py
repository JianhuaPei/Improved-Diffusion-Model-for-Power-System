
from munch import DefaultMunch


Config={
    #data_path
    #'data_path':'F:\\Renewable_generation_and_load_data\\load_data_process\\Regular_data\\SCADA_data_of_case_118_sample',
    #'data_path':'F:\\Renewable_generation_and_load_data\\load_data_process\\Python_based_simulation\\IEEE_39_bus_test_system\\ieee39_sample\\loadchanging', 
    'data_path':'F:\\Renewable_generation_and_load_data\\load_data_process\\Python_based_simulation\\NPCC_140_bus_test_system\\npcc140_sample_original', 

    # SCADA IEEE 30: 24*96 IEEE 57: 48*96 IEEE 118: 96*96  WAMS IEEE 39: 48*120 NPCC 140: 120*120
    'inputdata_size_M': 120,
    'inputdata_size_T': 120, 

    'batch_size_ddpm':2,
    'epochs_ddpm':20,
    'lr_ddpm':5e-4, #learning rate of ddpm
    
    'latent_dim':64,
    
    'timesteps_ddpm': 100,
    'timesteps_ddim': 20,
    'batch_size_GAN':32,
    'epochs_GAN':200,
    'lr_GAN':0.0001, #learning rate of GAN
    'latent_dim':64, #latent dimension
    'n_critic':5, #the number of training D when training G once
    'clip_value':0.01, # clip D
    #'load_G_name':'my_G300.pth', #load the model of G
    
    'epochs_Encoder':100,
    'batch_size_Encoder':8,
    
    'epochs_VAE':100,
    'batch_size_VAE':32,
    'lr_VAE':0.001,
    
    
    
    #'load_LSTM_name':'my_Lstm_E100000.pth', #load the model of lstm
    'epochs_lstm':100, #the training epochs of lstm
    'batch_size_lstm':32, #the bacth size of lstm
    'lr_lstm':0.01, #learning rate of lstm
    'J':12, #LSTM prediction lookback
    'hidden_size':128,  #hidden layer dimension
    "layer":2, #the number of layers



    }
dic_obj = DefaultMunch.fromDict(Config)

import os
import math
from abc import abstractmethod



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# use sinusoidal position embedding to encode time step
def timestep_embedding(timesteps, dim, max_period = 10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return an [N*dim] Tensor of positional embeddings.
    """
    half = dim//2
    freqs = torch.exp(-math.log(max_period)*torch.arange(start = 0, end = half,  dtype = torch.float32) / half).to(device = timesteps.device)
    args = timesteps[:,None].float()*freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:,:1])], dim=-1)      
    return embedding

# define TimestepEmbedSequential to support 'time_emb' as extra input
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    
    @abstractmethod
    def froward(self, x, emb):
        """
        Apply the module to 'x' given 'emb' timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it as extra input.
    """
    
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# use GN for norm layer
def norm_layer(channels):
    return nn.GroupNorm(32, channels)


# Residual block
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(norm_layer(in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
        #pojection for time step embedding
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))
        
        self.conv2 = nn.Sequential(norm_layer(out_channels),nn.SiLU(),nn.Dropout(p=dropout), nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1))
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, t):
        """
        'x' has shape [batch_size, in_dim, height, width]
        't' has shape [batch_size, time_dim]
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h+self.shortcut(x)
    
    
#Attention block with shortcut
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads = 1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels*3, kernel_size = 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size = 1)
        
    def forward(self, x):
        B,C,H,W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. /math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q*scale, k*scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts, bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h+x
    
#upsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv  = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
            
    def forward(self, x):
        x = F.interpolate(x, scale_factor = 2, mode= "nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

#downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size = 3, stride = 2, padding = 1)
        else:
            self.op = nn.AvgPool2d(stride = 2)
            
    def forward(self, x):
        return self.op(x)


# The full UNet model with attention and timestep embedding
class UNetModel(nn.Module):
    def __init__(self, in_channels = 1, model_channels = 256, out_channels = 1, num_res_blocks = 2, attention_resolutions = (8,16), dropout = 0, channel_mult = (1,2,2,2), conv_resample = True, num_heads = 4):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels =  model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        
        #time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim,time_embed_dim))
        
        #down blocks
        self.down_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size = 3, padding = 1))])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ ResidualBlock(ch, mult*model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads = num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1: #do not downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2
                
                
        # middle block
        self.middle_block = TimestepEmbedSequential(ResidualBlock(ch, ch, time_embed_dim, dropout), AttentionBlock(ch, num_heads = num_heads), ResidualBlock(ch,ch,time_embed_dim,dropout))
        
        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResidualBlock(ch+down_block_chans.pop(), model_channels*mult, time_embed_dim, dropout)]
                
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads = num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))
        
        self.out = nn.Sequential(norm_layer(ch), nn.SiLU(), nn.Conv2d(model_channels, out_channels, kernel_size = 3, padding = 1))
        
    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [ N * C * H * W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return an [N * C * ...] Tensor of outputs.
        """
        hs = []
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        #dowm stage
        h = x
        
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
            
        #middle stage
        h = self.middle_block(h,emb)

        #up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)
    
    
    # beta schedule
def linear_beta_schedule(timesteps):
    scale = 1000/timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x/timesteps)+s)/(1+s)*math.pi*0.5)**2
    alphas_cumprod = alphas_cumprod/alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion:
    def __init__(self, timesteps = 1000, beta_schedule = 'linear'):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
        
        self.alphas = 1. - self.betas # α_n = 1 - β_n
        self.alphas_cumprod  =torch.cumprod(self.alphas, axis=0) # cumprod of α_n
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],(1,0),value=1.) # cumprod of α_{n-1}
        
        
        #calculations for diffusion q(x_n / x_{n-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) #sqrt{α_n}
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod) #sqrt{1- α_n}
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod) # log(1 - α_n )
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod) # sqrt{1/α_n}
        self.sqrt_recipml_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1) #1/sqrt{1/α_n - 1}
        
        #calculation for posterior q(x_{n-1}/x_n, x_0)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev)/ (1.0 - self.alphas_cumprod)) #(1-α_{n-1})*β_n/{1- α_n}
        #below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # model_var_type -> fixedsmall
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min = 1e-20)) 
        # model_var_type -> fixedlarge
        self.posterior_log_variance_clipped_ddim = torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev)/(1.0 - self.alphas_cumprod)) # sqrt{α_{n-1}}*β_n/{1-α_n}
        
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev)*torch.sqrt(self.alphas)/(1.0 - self.alphas_cumprod)) #sqrt{α_n}(1 - α_{n-1} )/{1-α_n}

        
    #get the param of given timestep n
    def _extract(self, a, n, x_shape):
        batch_size = n.shape[0]
        out = a.to(n.device).gather(0,n).float()
        out = out.reshape(batch_size, *((1,)*(len(x_shape)-1)))
        return out
        
    # forward diffusion (using the nice property): q(x_n/ x_0) 
    def q_sample(self, x_start, n, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
                
        sqrt_alphas_cumprod_n = self._extract(self.sqrt_alphas_cumprod, n, x_start.shape) #sqrt{α_n}
        sqrt_one_minus_alphas_cumprod_n = self._extract(self.sqrt_one_minus_alphas_cumprod, n, x_start.shape) #sqrt{1- α_n}
        
        x_noisy = sqrt_alphas_cumprod_n * x_start + sqrt_one_minus_alphas_cumprod_n * noise
            
        return  x_noisy # sqrt{α_n}*x_0 + sqrt{1- α_n}*noise
        
    #Get the mean and variance of q(x_n/ x-0).
    def q_mean_variance(self, x_start, n):
        mean = self._extract(self.sqrt_alphas_cumprod,n,x_start.shape)*x_start # sqrt{α_n}*x_0
        variance = self._extract(1.0 - self.alphas_cumprod,n,x_start.shape) # 1- α_n
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, n, x_start.shape)  #log(1 - α_n )
        return mean, variance, log_variance
        
    #Compute the mean and variance of the diffusion posterior: q(x_{n-1}) / x_n, x_0)
    def q_posterior_mean_variance(self, x_start, x_n, n):
        posterior_mean = (self._extract(self.posterior_mean_coef1, n, x_n.shape)*x_start  + self._extract(self.posterior_mean_coef2,n,x_n.shape)*x_n)
        posterior_variance = self._extract(self.posterior_variance, n, x_n.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, n, x_n.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        
    #compute x_0 from x_t and pred noise: the reverse of 'q_sample'
    def predict_start_from_noise(self, x_n, n, noise):
        return (self._extract(self.sqrt_recip_alphas_cumprod, n, x_n.shape)*x_n - self._extract(self.sqrt_recipml_alphas_cumprod, n, x_n.shape)*noise)     
        
    #compute predicted mean and variance of p(x_{n-1} / x_n)
    def p_mean_variance(self, model, x_n, n, clip_denoised = True):
        # predict noise using model
        print(x_n.shape)
        pred_noise = model(x_n,n) 
        #get the predicted x_0: 
        x_recon = self.predict_start_from_noise(x_n, n, pred_noise)
        if clip_denoised: 
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_n, n)
        return model_mean, posterior_variance, posterior_log_variance
        
    #denoise_step: sample x_{n-1} from x_n and pred_noise
    @torch.no_grad()
    def p_sample(self, model, x_n, n, clip_denoised=True):
        #predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_n, n, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_n)
        # no noise when n == 0
        nonzero_mask = ((n!=0).float().view(-1, *([1]*(len(x_n.shape) - 1))))
        #compute x_{n-1}
        pred_measurements = model_mean + nonzero_mask * (0.5*model_log_variance).exp()*noise
        return pred_measurements
        
    #denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        measurement = torch.randn(shape, device = device)
        measurements = []
        for i in tqdm(reversed(range(0,self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            measurement = self.p_sample(model, measurement, torch.full((batch_size,), i, device= device, dtype=torch.long))
            measurements.append(measurement.cpu().numpy())
        return measurements
     

    
    #sample new images
    @torch.no_grad()
    def sample(self, model, input_size_M, input_size_T, batch_size = 1, channels = 1):
        return self.p_sample_loop(model, shape = (batch_size, channels, input_size_M, input_size_T))
    
    #use ddim to sample
    @torch.no_grad()
    def ddim_sample(self, model, input_size_M, input_size_T, batch_size = 1, channels = 1, ddim_timesteps=50, ddim_discr_method='uniform', ddim_eta=0.0, clip_denoised=True):
        #make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0,self.timesteps,c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = ((np.linspace(0, np.sqrt(self.timesteps * .8),ddim_timesteps))**2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        
        #previous sequence
        ddim_timestep_prev_seq =  np.append(np.array([0]), ddim_timestep_seq[:-1])
        device = next(model.parameters()).device
        #start from pure noise (for each example in the batch)
        sample_measurement = torch.randn((batch_size, channels, input_size_M, input_size_T), device=device)
        for i in tqdm(reversed(range(0,ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            n = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_n = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
                        
            # 1. get current and previous alphas_cumprod
            alpha_cumprod_n = self._extract(self.alphas_cumprod, n, sample_measurement.shape)
            alpha_cumprod_n_prev = self._extract(self.alphas_cumprod, prev_n, sample_measurement.shape)
            
            # 2. predict noise using model
            pred_noise = model(sample_measurement, n)
            
            # 3. get the predicted x_0
            pred_x0 = (sample_measurement -torch.sqrt((1. - alpha_cumprod_n))*pred_noise) / torch.sqrt(alpha_cumprod_n)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
                
            # 4. compute variance: "sigma_n"
            # σ_n = sqrt((1 − α_n−1)/(1 − α_n)) * sqrt(1 − α_n/α_n−1)            
            sigmas_n = ddim_eta * torch.sqrt((1-alpha_cumprod_n_prev)/(1 - alpha_cumprod_n)*(1 - alpha_cumprod_n / alpha_cumprod_n_prev))
            
            
            # 5. compute "direction pointing to x_n"
            
            pred_dir_xn = torch.sqrt(1 - alpha_cumprod_n_prev - sigmas_n**2)*pred_noise
            
            # 6. compute x_{n-1}
            x_prev = torch.sqrt(alpha_cumprod_n_prev)* pred_x0 +pred_dir_xn + sigmas_n*torch.randn_like(sample_measurement)
            
            sample_measurement = x_prev
            
        return sample_measurement.cpu().numpy() 
    
    
    
    #use ddim with optimal variance to sample
    @torch.no_grad()
    def ddim_sample_with_optimal_variance(self, model, input_size_M, input_size_T, batch_size = 1, channels = 1, ddim_timesteps = 50, ddim_discr_method='uniform', ddim_eta = 0.0, ms_eps = None , clip_denoised = True):
        #make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0,self.timesteps,c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = ((np.linspace(0, np.sqrt(self.timesteps * .8),ddim_timesteps))**2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        
        if ms_eps is None:
            ms_eps = np.zeros( self.timesteps, dtype=np.float32)
        ms_eps = torch.tensor(ms_eps)
        #previous sequence
        ddim_timestep_prev_seq =  np.append(np.array([0]), ddim_timestep_seq[:-1])
        device = next(model.parameters()).device
        #start from pure noise (for each example in the batch)
        x_tau_i = torch.randn((batch_size, channels, input_size_M, input_size_T), device=device)
        noise_para = 1.0
        for i in tqdm(reversed(range(0,ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            tau_i = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_tau_i = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
                        
            # 1. get current and previous alphas_cumprod
            alpha_cumprod_n = self._extract(self.alphas_cumprod, tau_i, x_tau_i.shape)
            alpha_cumprod_n_prev = self._extract(self.alphas_cumprod, prev_tau_i, x_tau_i.shape)
            ms_eps_n = self._extract(ms_eps, tau_i, x_tau_i.shape)
            
            # 2. predict noise using model
            pred_noise = model(x_tau_i, tau_i)
            
            # 3. get the predicted x_0
            pred_x0 = (x_tau_i -torch.sqrt((1. - alpha_cumprod_n))*pred_noise) / torch.sqrt(alpha_cumprod_n)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
                
            # 4. compute optimal_variance, original_variance, theta, gamma
            sigmas_n = ddim_eta * torch.sqrt((1-alpha_cumprod_n_prev)/(1. - alpha_cumprod_n)*(1. - alpha_cumprod_n / alpha_cumprod_n_prev))
            sigmas_n_bar_2 = (1-alpha_cumprod_n) / alpha_cumprod_n * (1. - ms_eps_n)
            sigmas_n_bar_2 = torch.clamp(sigmas_n_bar_2, min=0., max=1.)
            theta = torch.sqrt((1 - alpha_cumprod_n_prev - sigmas_n**2)/(1-alpha_cumprod_n)) 
            gamma = torch.sqrt(alpha_cumprod_n_prev) - theta * torch.sqrt( alpha_cumprod_n)    
            
            # 5. compute  x_{n-1}  
            tau_i_np = tau_i.numpy()
            if tau_i_np.any() == 1:
                noise_para = 0.0
            x_prev = theta * x_tau_i + gamma * pred_x0 + noise_para * torch.sqrt(sigmas_n**2 + (gamma**2)*(sigmas_n_bar_2))*torch.randn_like(x_tau_i)
            
            x_tau_i = x_prev
            
        return x_tau_i.cpu().numpy()             


    # conditioned ddim
    @torch.no_grad()
    def conditioned_ddim(self, model, input_size_M, input_size_T, batch_size = 1, channels = 1, ddim_timesteps = 50, ddim_discr_method = 'uniform', ddim_eta = 0.0, ms_eps = None, y_start = None, omega =0.0, clip_denoised=True):
        #make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0,self.timesteps,c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = ((np.linspace(0, np.sqrt(self.timesteps * .8),ddim_timesteps))**2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        if ms_eps is None:
            ms_eps = np.zeros( self.timesteps, dtype=np.float32)
        ms_eps = torch.tensor(ms_eps)
        
        #previous sequence
        ddim_timestep_prev_seq =  np.append(np.array([0]), ddim_timestep_seq[:-1])
        device = next(model.parameters()).device

        x_tau_i = torch.randn((batch_size, channels, input_size_M, input_size_T), device=device)
        noise_para = 1.0
        
        output_list = []
        output_tau_list = []
        list_show = [1,int(ddim_timesteps/2),ddim_timesteps-4,ddim_timesteps]
        ii = 0
        
        for i in tqdm(reversed(range(0,ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            tau_i = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)        
            prev_tau_i = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long) 

            # 1. get current and previous alphas_cumprod
            alpha_cumprod_n = self._extract(self.alphas_cumprod, tau_i, x_tau_i.shape)
            alpha_cumprod_n_prev = self._extract(self.alphas_cumprod, prev_tau_i, x_tau_i.shape)
            ms_eps_n = self._extract(ms_eps, tau_i, x_tau_i.shape)
            
            # 2. predict noise using model
            pred_noise = model(x_tau_i, tau_i)   
            
            # 3. compute y_tau_i and corrected noise term
            y_tau_i =  torch.sqrt(alpha_cumprod_n) * y_start + torch.sqrt(1. - alpha_cumprod_n) * pred_noise
            #y_tau_i =  torch.sqrt(alpha_cumprod_n) * y_start + torch.sqrt(1. - alpha_cumprod_n) * torch.randn_like(x_tau_i)
            corrected_noise = pred_noise -  omega * torch.sqrt(1. - alpha_cumprod_n) * (y_tau_i - x_tau_i) 
            
            # 4. get the predicted x_0
            pred_x0 = (x_tau_i -torch.sqrt((1. - alpha_cumprod_n))*corrected_noise) / torch.sqrt(alpha_cumprod_n)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)  
                
            # 5. compute optimal_variance, original_variance, theta, gamma
            sigmas_n = ddim_eta * torch.sqrt((1-alpha_cumprod_n_prev)/(1. - alpha_cumprod_n)*(1. - alpha_cumprod_n / alpha_cumprod_n_prev))
            sigmas_n_bar_2 = (1-alpha_cumprod_n) / alpha_cumprod_n * (1. - ms_eps_n)
            sigmas_n_bar_2 = torch.clamp(sigmas_n_bar_2, min=0., max=1.)
            theta = torch.sqrt((1 - alpha_cumprod_n_prev - sigmas_n**2)/(1-alpha_cumprod_n)) 
            gamma = torch.sqrt(alpha_cumprod_n_prev) - theta * torch.sqrt( alpha_cumprod_n)   
            
            # 6. compute  x_{n-1}  
            tau_i_np = tau_i.numpy()

            if tau_i_np == 1:
                noise_para = 0.0
            x_prev = theta * x_tau_i + gamma * pred_x0 + torch.sqrt(sigmas_n**2 + noise_para*(gamma**2)*(sigmas_n_bar_2))*torch.randn_like(x_tau_i)
            
            x_tau_i = x_prev
            
            ii = ii + 1
            if ii in list_show:
                output_tau_list.append(tau_i_np)
                output_list.append(x_tau_i.cpu().numpy() ) 
            
        return output_tau_list, output_list               

    # ddim imputation
    @torch.no_grad()
    def ddim_imputation(self, model, input_size_M, input_size_T, batch_size = 1, channels = 1, ddim_timesteps = 50, ddim_discr_method = 'uniform', ddim_eta = 0.0, ms_eps = None, y_start = None, R = 1,  clip_denoised=True):                 
        #make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0,self.timesteps,c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = ((np.linspace(0, np.sqrt(self.timesteps * .8),ddim_timesteps))**2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        
        if ms_eps is None:
            ms_eps = np.zeros( self.timesteps, dtype=np.float32)
        ms_eps = torch.tensor(ms_eps)
        
        y_zeros = torch.zeros_like(y_start)
        y_ones = torch.ones_like(y_start)
        y_omega = torch.where(torch.isnan(y_start), y_zeros, y_start)
        y_nan = torch.where(torch.isnan(y_start), y_ones, y_zeros)
        y_non_nan = torch.where(torch.isnan(y_start), y_zeros, y_ones)
        
        
        #previous sequence
        ddim_timestep_prev_seq =  np.append(np.array([0]), ddim_timestep_seq[:-1])
        device = next(model.parameters()).device

        x_tau_i = torch.randn((batch_size, channels, input_size_M, input_size_T), device=device)
        noise_para = 1.0 
        output_list = []
        output_tau_list = []
        list_show = [1,int(ddim_timesteps/2),ddim_timesteps-4,ddim_timesteps]
        ii = 0       
        for i in tqdm(reversed(range(0,ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            tau_i = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)        
            prev_tau_i = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long) 
            
            # 1. get current and previous alphas_cumprod
            alpha_cumprod_n = self._extract(self.alphas_cumprod, tau_i, x_tau_i.shape)
            alpha_cumprod_n_prev = self._extract(self.alphas_cumprod, prev_tau_i, x_tau_i.shape)
            betas_n = self._extract(self.betas, tau_i, x_tau_i.shape)
            ms_eps_n = self._extract(ms_eps, tau_i, x_tau_i.shape)  
            tau_i_numpy = tau_i.numpy()

            
            if tau_i_numpy > 1:
                        
                for r in range(R):
                    # 2. compute noisy data of known tau_{i-1}
                    x_omega_tau_i_prev = torch.sqrt(alpha_cumprod_n_prev)*y_omega + torch.sqrt(1. - alpha_cumprod_n_prev)*torch.randn_like(y_omega)
                
                    # 3. predict noise using model
                    pred_noise = model(x_tau_i, tau_i)
            
                    # 4. get the predicted x_0
                    pred_x0 = (x_tau_i -torch.sqrt((1. - alpha_cumprod_n))*pred_noise) / torch.sqrt(alpha_cumprod_n)
                    if clip_denoised:
                        pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
                
                    # 5. compute optimal_variance, original_variance, theta, gamma
                    sigmas_n = ddim_eta * torch.sqrt((1-alpha_cumprod_n_prev)/(1. - alpha_cumprod_n)*(1. - alpha_cumprod_n / alpha_cumprod_n_prev))
                    sigmas_n_bar_2 = (1-alpha_cumprod_n) / alpha_cumprod_n * (1. - ms_eps_n)
                    sigmas_n_bar_2 = torch.clamp(sigmas_n_bar_2, min=0., max=1.)
                    theta = torch.sqrt((1 - alpha_cumprod_n_prev - sigmas_n**2)/(1-alpha_cumprod_n)) 
                    gamma = torch.sqrt(alpha_cumprod_n_prev) - theta * torch.sqrt( alpha_cumprod_n)    
            
                    # 6. compute  x_{tau_i-1}  
                    x_non_omega_tau_i_prev = theta * x_tau_i + gamma * pred_x0 + torch.sqrt(sigmas_n**2 + (gamma**2)*(sigmas_n_bar_2))*torch.randn_like(x_tau_i)
                    x_tau_i_prev = x_omega_tau_i_prev * y_non_nan + x_non_omega_tau_i_prev * y_nan
                
                    # 7. resampling
                    if r < (R-1):
                        x_tau_i = torch.sqrt(1. - betas_n)*x_tau_i_prev + torch.sqrt(betas_n)*torch.randn_like(x_tau_i)
                    
                x_tau_i = x_tau_i_prev
            
            if tau_i_numpy == 1:
                x_omega_tau_i_prev = torch.sqrt(alpha_cumprod_n_prev)*y_omega
                pred_noise = model(x_tau_i, tau_i)
                pred_x0 = (x_tau_i -torch.sqrt((1. - alpha_cumprod_n))*pred_noise) / torch.sqrt(alpha_cumprod_n)
                if clip_denoised:
                    pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
                    
                sigmas_n = ddim_eta * torch.sqrt((1-alpha_cumprod_n_prev)/(1. - alpha_cumprod_n)*(1. - alpha_cumprod_n / alpha_cumprod_n_prev))
                sigmas_n_bar_2 = (1-alpha_cumprod_n) / alpha_cumprod_n * (1. - ms_eps_n)
                sigmas_n_bar_2 = torch.clamp(sigmas_n_bar_2, min=0., max=1.)
                theta = torch.sqrt((1 - alpha_cumprod_n_prev - sigmas_n**2)/(1-alpha_cumprod_n)) 
                gamma = torch.sqrt(alpha_cumprod_n_prev) - theta * torch.sqrt( alpha_cumprod_n)  
                x_non_omega_tau_i_prev = theta * x_tau_i + gamma * pred_x0
                x_tau_i_prev = x_omega_tau_i_prev * y_non_nan + x_non_omega_tau_i_prev * y_nan
                
                x_tau_i = x_tau_i_prev
            
            ii = ii + 1
            if ii in list_show:
                output_tau_list.append(tau_i_numpy)
                output_list.append(x_tau_i.cpu().numpy()  )
            
        return output_tau_list, output_list
        
                
                
    # sample new images
    @torch.no_grad()
    def sample(self, model, input_size_M, input_size_T, batch_size = 1, channels = 1):
        return self.p_sample_loop(model, shape=(batch_size, channels, input_size_M, input_size_T))
        
    # compute train losses
    def train_losses(self, model, x_start, n):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_n
        x_noisy = self.q_sample(x_start, n, noise = noise)
        predicted_noise = model(x_noisy, n)
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    
    


    
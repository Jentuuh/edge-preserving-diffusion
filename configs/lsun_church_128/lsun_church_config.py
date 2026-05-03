import ml_collections
import torch
import numpy as np
import decimal

# TODO: clean up!

def get_config():
    return get_default_configs()

def ceil(n, d):
    return int(-(n // -d))

def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 32
    training.num_epochs = 180
    training.sample_freq = 10000
    training.checkpoint_freq = 10000

    # Enables loading from checkpoint 
    config.resume = True

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'church_res256'
    data.image_size = 256
    data.num_channels = 3

    # diffusion
    config.diffusion = diffusion = ml_collections.ConfigDict()
    diffusion.num_steps = 1000
    diffusion.beta_min = 1e-4
    diffusion.beta_max = 2e-2
    diffusion.lambda_min = 1e-6
    diffusion.lambda_max = 1e-1
    diffusion.transition_pt = 0.5
    diffusion.k = 20

    # model
    config.model = model = ml_collections.ConfigDict()
    model.dropout = 0.1
    model.model_channels = 128  # Base amount of channels in the model
    model.channel_mult = (1, 2, 2, 2)
    model.conv_resample = True
    model.num_heads = 1
    model.conditional = True
    model.attention_levels = (4,)
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.num_res_blocks = 2
    model.use_fp16 = False
    model.use_scale_shift_norm = False
    model.resblock_updown = False
    model.use_new_attention_order = True
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.skip_rescale = True

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.lr = 2e-5
    optim.dim_lr = False
    optim.dim_lr_freq = 200 # Amount of epochs after which LR will be diminished
    optim.min_lr = 2e-5
    
    config.device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config

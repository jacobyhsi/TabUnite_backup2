import time
import json
import torch
import numpy as np
import pandas as pd

from dataset import SynthDataset
from methods.tabddpm.models.modules import MLPDiffusion
from methods.tabddpm.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion

def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
): 
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

def sample(
    model_save_path,
    sample_save_path,
    real_data_path,
    dataname,
    steps = 1000,
    batch_size = 1024,
    model_type='mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    device=torch.device('cuda:0'),
    num_samples = 0,
    num_numerical_features = 0,
):
    dataset = SynthDataset(dataname)

    K = np.array(dataset.get_category_sizes())
    num_numerical_features = dataset.get_numerical_sizes()

    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in

    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=K
    )

    model_path =f'{model_save_path}/model.pt'
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )
    diffusion.to(device)
    diffusion.eval()

    start_time = time.time()

    x_gen = diffusion.sample_all(num_samples, batch_size, ddim=False, steps = steps)
    syn_df = x_gen
    
    end_time = time.time()

    print('Sampling time:', end_time - start_time)

    save_path = sample_save_path
    
    syn_df = pd.DataFrame(syn_df)
    
    syn_df.iloc[:, :num_numerical_features] = dataset.quantile_scaler.inverse_transform(syn_df.iloc[:, :num_numerical_features])
    
    info_path = f'{real_data_path}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)
        
    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    
    syn_df.to_csv(save_path, index = False)
    print('Saving sampled data to {}'.format(save_path))
    
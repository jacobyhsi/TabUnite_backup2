import torch
import numpy as np
import pandas as pd
import os
import json
import time

from dataset import SynthDataset
from methods.oheddpm.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion
from methods.oheddpm.models.modules import MLPDiffusion

import src
from utils_train import make_dataset

@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse):
    task_type = info['task_type']

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]

    syn_num = num_inverse(syn_num).astype(np.float32)
    syn_cat = cat_inverse(syn_cat)


    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    
    else:
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target

def recover_data(syn_num, syn_cat, syn_target, info):

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]


    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df

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

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def sample(
    model_save_path,
    sample_save_path,
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

    K = dataset.get_category_sizes()
    num_numerical_features = dataset.get_numerical_sizes()
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = int(d_in)
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
        np.array(K),
        num_numerical_features=num_numerical_features,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()

    start_time = time.time()
    #########

    x_gen = diffusion.sample_all(num_samples, batch_size, ddim=False, steps = steps)
    syn_df = x_gen
    
    ##########
    
    # final_data = []
    # batch_size = 2458285 // 10  # Calculated batch size

    # for i in range(10):  # Adjusted for 5 batches
    #     batch_samples = cfm_sampler(model, batch_size, d_in, N=50, device=device, use_tqdm=True)
    #     final_data.append(batch_samples)

    # # Concatenate the list of tensors into a single tensor
    # syn_df = np.concatenate(final_data, axis=0)
    # print("syn_df.shape", syn_df.shape)
    
    ##########
    
    end_time = time.time()

    print('Sampling time:', end_time - start_time)

    save_path = sample_save_path
    
    syn_df = pd.DataFrame(syn_df)
    syn_df.iloc[:, :num_numerical_features] = dataset.quantile_scaler.inverse_transform(syn_df.iloc[:, :num_numerical_features])
    syn_df.to_csv(save_path, index = False)
    print('Saving sampled data to {}'.format(save_path))
    
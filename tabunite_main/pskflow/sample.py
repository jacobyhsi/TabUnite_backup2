import time
import json
import torch
import numpy as np
import pandas as pd

import src
from utils_train import make_dataset
from pskflow.models.flow_matching import sample as cfm_sampler
from pskflow.models.modules import MLPDiffusion, Model
from pskflow.models.flow_matching import ConditionalFlowMatcher

def bits_needed(categories):
    return 2 * np.ones_like(categories)

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
        print(syn_cat.shape)
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


def sample(
    model_save_path,
    sample_save_path,
    real_data_path,
    batch_size = 2000,
    num_samples = 0,
    task_type = 'binclass',
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:0'),
    change_val = False,
    ddim = False,
    steps = 1000,
):
    emb_dim = 3

    T = src.Transformations(**T_dict)

    D = make_dataset(
        real_data_path,
        T,
        task_type = task_type,
        change_val = False,
    )

    K = np.array(D.get_category_sizes('train'))
    num_numerical_features = D.X_num['train'].shape[1] if D.X_num is not None else 0
    num_bits_per_cat_feature = bits_needed(K) if len(K) > 0 else np.array([0])
    d_in = np.sum(num_bits_per_cat_feature) + num_numerical_features
    model_params['d_in'] = d_in

    flow_net = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=D.get_category_sizes('train')
    )

    model_path =f'{model_save_path}/model.pt'
    flow_net.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    cfm = ConditionalFlowMatcher(sigma=0.0, pred_x1=False)
    model = Model(
        flow_net,
        cfm,
        num_numerical_features,
        K,
        num_bits_per_cat_feature
    )
    model.to(device)
    model.eval()

    start_time = time.time()

    x_gen = cfm_sampler(model, num_samples, d_in, N=50, device=device, use_tqdm=True) # Sampling Steps
    syn_data = x_gen
    ############################################################### SAMPLE IS CHANGED HERE FOR LARGE DATASET ##########################################
    # final_data = []
    # batch_size = 2458285 // 5  # Calculated batch size

    # for i in range(5):  # Adjusted for 5 batches
    #     batch_samples = cfm_sampler(model, batch_size, d_in, N=50, device=device, use_tqdm=True)
    #     final_data.append(batch_samples)

    # # Concatenate the list of tensors into a single tensor
    # syn_data = np.concatenate(final_data, axis=0)
    # print("syn_data.shape", syn_data.shape)
    ####################################################################################################################################################
    
    num_inverse = D.num_transform.inverse_transform
    cat_inverse = D.cat_transform.inverse_transform
    
    info_path = f'{real_data_path}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse) 
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    end_time = time.time()

    print('Sampling time:', end_time - start_time)

    save_path = sample_save_path
    syn_df.to_csv(save_path, index = False)
    print('Saving sampled data to {}'.format(save_path))

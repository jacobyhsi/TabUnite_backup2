import time
import json
import torch
import numpy as np
import pandas as pd

from dataset import SynthDataset
from methods.oheflow.models.flow_matching import sample as cfm_sampler
from methods.oheflow.models.modules import MLPDiffusion, Model
from methods.oheflow.models.flow_matching import ConditionalFlowMatcher

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
    dataname,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    num_numerical_features = 0,
    device = torch.device('cuda:0'),
):

    dataset = SynthDataset(dataname)

    K = dataset.get_category_sizes()
    num_numerical_features = dataset.get_numerical_sizes()
    d_in = np.sum(np.array(K)) + num_numerical_features
    model_params['d_in'] = d_in

    flow_net = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=K
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
        np.array(K)
    )
    model.to(device)
    model.eval()

    start_time = time.time()

    if num_samples < 100000:
        x_gen = cfm_sampler(model, num_samples, d_in, N=50, device=device, use_tqdm=True)
        syn_df = x_gen
    else:
        final_data = []
        batch_size = num_samples // 17  # Calculated batch size

        for i in range(17):  # Adjusted for 5 batches
            batch_samples = cfm_sampler(model, batch_size, d_in, N=50, device=device, use_tqdm=True)
            final_data.append(batch_samples)

        # Concatenate the list of tensors into a single tensor
        syn_df = np.concatenate(final_data, axis=0)
        print("syn_df.shape", syn_df.shape)
    
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
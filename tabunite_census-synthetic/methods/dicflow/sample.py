import time
import json
import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer
from dataset import SynthDataset
from methods.dicflow.models.modules import MLPDiffusion, Model
from methods.dicflow.models.flow_matching import ConditionalFlowMatcher
from methods.dicflow.models.flow_matching import sample as cfm_sampler

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
    real_data_path,
    dataname,
    sample_save_path,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    num_numerical_features = 0,
    device = torch.device('cuda:0'),
):
    dataset = SynthDataset(dataname)

    K = dataset.get_category_sizes()
    num_numerical_features = dataset.get_numerical_sizes()
    
    emb_dim = 1
    cat_len = 0
    for item in np.array(K):
        cat_len += 1 if item <=2 else emb_dim

    d_in = num_numerical_features + cat_len
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
        np.array(K),
        emb_dim
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

        for i in range(17):  # Adjusted for 17 batches
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
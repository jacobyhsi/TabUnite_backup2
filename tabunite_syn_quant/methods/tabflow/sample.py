import torch
import numpy as np

from dataset import OnlineToyDataset
from methods.tabflow.models.modules import MLPDiffusion
from methods.tabflow.models.mixed_flow import ContinuousDiscreteFlow

def bits_needed(categories):
    return np.ceil(np.log2(categories)).astype(int)

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
    dataname,
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type='mlp',
    model_params = None,
    device=torch.device('cuda:0'),
):
    dataset = OnlineToyDataset(dataname)

    K = np.array(dataset.get_category_sizes())
    num_numerical_features = dataset.get_numerical_sizes()
    
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in

    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes()
    )
    
    model_path =f'{model_save_path}/model.pt'
    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    cfm = ContinuousDiscreteFlow(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        flow_net=model,
        device=device
    )
    cfm.to(device)
    cfm.eval()

    num_samples = 20000

    step = 1000
    x_gen = cfm.mixed_sample(num_samples, N=step, device=device, use_tqdm=True)
    acc = dataset.evaluate(x_gen)
    print(f'Accuracy: {acc}')


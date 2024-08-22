import torch
import numpy as np

from dataset import OnlineToyDataset
from methods.tabddim.models.modules import MLPDiffusion
from methods.tabddim.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion

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
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
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

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )
    diffusion.to(device)
    diffusion.eval()

    num_samples = 20000

    step = 1000
    x_gen = diffusion.sample_all(num_samples, batch_size, ddim=True, steps=step)
    acc = dataset.evaluate(x_gen.cpu().detach().numpy())
    print(f'Accuracy: {acc}')


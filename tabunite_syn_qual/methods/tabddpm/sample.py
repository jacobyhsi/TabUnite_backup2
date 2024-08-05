import time
import torch
import numpy as np

from dataset import OnlineToyDataset
from methods.tabddpm.models.modules import MLPDiffusion
from methods.tabddpm.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion
from dataset import plot_rings_example, plot_25_gaussian_example, plot_25_circles_example, plot_olympic_example

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

    ddim = False
    num_samples = 20000

    # for steps in [2, 5, 10, 30, 50, 100, 300, 500, 1000]:
    for steps in [2, 10, 100, 500]:
        print("sampling steps: ", steps)
        if not ddim:
            x_gen = diffusion.sample_all(num_samples, batch_size, ddim=False, steps = steps)
        else:
            x_gen = diffusion.sample_all(num_samples, batch_size, ddim=True, steps = steps)
        x_gen = x_gen.cpu().numpy()

        if dataname == 'rings':
            plot_rings_example(x_gen, f'{model_save_path}/rings_tabddpm_steps={steps}.png')
        elif dataname == 'olympic':
            plot_olympic_example(x_gen, f'{model_save_path}/olympic_tabddpm_steps={steps}.png')
        elif dataname == '25gaussians':
            plot_25_gaussian_example(x_gen, f'{model_save_path}/25gaussians_dilflow_steps={steps}.png')
        elif dataname == '25circles':
            plot_25_circles_example(x_gen, f'{model_save_path}/25circles_dilflow_steps={steps}.png')
        else:
            raise "Unknown dataset!"

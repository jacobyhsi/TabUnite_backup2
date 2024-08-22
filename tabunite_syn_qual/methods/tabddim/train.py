import os
import time
import torch
import numpy as np
import pandas as pd
from copy import deepcopy

from dataset import OnlineToyDataset
from methods.tabddim.models.modules import MLPDiffusion
from methods.tabddim.models.gaussian_multinomial_distribution import GaussianMultinomialDiffusion

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

class Trainer:
    def __init__(self, diffusion, dataset, batch_size, lr, weight_decay, steps, model_save_path, device=torch.device('cuda:1')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.dataset = dataset
        self.batch_size = batch_size
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.model_save_path = model_save_path

        columns = list(np.arange(5)*200)
        columns[0] = 1
        columns = ['step'] + columns

        self.log_every = 50
        self.print_every = 1
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x):
        x = x.to(self.device)

        self.optimizer.zero_grad()

        loss_multi, loss_gauss = self.diffusion.mixed_loss(x)

        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        self.print_every = 1
        self.log_every = 1

        best_loss = np.inf
        print('Steps: ', self.steps)
        while step < self.steps:
            start_time = time.time()
            x = self.dataset.gen_batch(self.batch_size)
            x = torch.tensor(x, dtype=torch.float32)
            
            batch_loss_multi, batch_loss_gauss = self._run_step(x)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            
            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if np.isnan(gloss):
                    print('Finding Nan')
                    break
                
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]

                np.set_printoptions(suppress=True)
          
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

                if mloss + gloss < best_loss:
                    best_loss = mloss + gloss
                    torch.save(self.diffusion._denoise_fn.state_dict(), os.path.join(self.model_save_path, 'model.pt'))
  
                if (step + 1) % 10000 == 0:
                    torch.save(self.diffusion._denoise_fn.state_dict(), os.path.join(self.model_save_path, f'model_{step+1}.pt'))

            # update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1
            # end_time = time.time()
            # print('Time: ', end_time - start_time)

def train(
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
    model.to(device)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        dataset,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        model_save_path=model_save_path,
        device=device
    )
    trainer.run_loop()

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(model_save_path, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(model_save_path, 'model_ema.pt'))

    trainer.loss_history.to_csv(os.path.join(model_save_path, 'loss.csv'), index=False)

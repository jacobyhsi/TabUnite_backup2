import os
import time
import torch
import numpy as np
import pandas as pd
from copy import deepcopy

from dataset import SynthDataset
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

class Trainer:
    def __init__(self, cfm_model, dataset, batch_size, lr, weight_decay, steps, model_save_path, device=torch.device('cuda:1')):
        self.model = cfm_model
        self.ema_model = deepcopy(self.model.flow_net)
        for param in self.ema_model.parameters():
            param.detach_()

        self.dataset = dataset
        self.batch_size = batch_size
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'loss'])
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

        loss = self.model.mixed_loss(x)

        loss.backward()
        self.optimizer.step()

        return loss


    def run_loop(self):
        step = 0
        curr_loss = 0.0

        curr_count = 0
        self.print_every = 1
        self.log_every = 1

        best_loss = np.inf
        print('Steps: ', self.steps)
        while step < self.steps:
            start_time = time.time()
            x = self.dataset.gen_batch(self.batch_size)
            x = torch.tensor(x, dtype=torch.float32)
            
            batch_loss = self._run_step(x)

            # self._anneal_lr(step)

            curr_count += len(x)
            curr_loss += batch_loss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                loss = np.around(curr_loss / curr_count, 4)
                if np.isnan(loss):
                    print('Finding Nan')
                    break
                
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} Loss: {loss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, loss]

                np.set_printoptions(suppress=True)
          
                curr_count = 0
                curr_loss = 0.0

                if loss < best_loss:
                    best_loss = loss
                    torch.save(self.model.flow_net.state_dict(), os.path.join(self.model_save_path, 'model.pt'))
  
                if (step + 1) % 10000 == 0:
                    torch.save(self.model.flow_net.state_dict(), os.path.join(self.model_save_path, f'model_{step+1}.pt'))

            # update_ema(self.ema_model.parameters(), self.model.flow_net.parameters())

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
    model_type = 'mlp',
    model_params = None,
    device = torch.device('cuda:0'),
):

    dataset = SynthDataset(dataname)
    K = dataset.get_category_sizes()
    num_numerical_features = dataset.get_numerical_sizes()

    d_in = np.sum(np.array(K)) + num_numerical_features
    model_params['d_in'] = d_in

    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=K
    )
    model.to(device)

    cfm = ContinuousDiscreteFlow(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        flow_net=model,
        device=device
    )

    trainer = Trainer(
        cfm,
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

    torch.save(cfm.flow_net.state_dict(), os.path.join(model_save_path, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(model_save_path, 'model_ema.pt'))

    trainer.loss_history.to_csv(os.path.join(model_save_path, 'loss.csv'), index=False)

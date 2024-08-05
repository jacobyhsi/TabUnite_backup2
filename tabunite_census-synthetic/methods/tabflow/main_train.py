import os

import src
from methods.tabflow.train import train

def main(args):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    args.train = True
    raw_config = src.load_config(config_path)

    ''' 
    Modification of configs
    '''
    print('START TRAINING TabFlow')

    train(
        **raw_config['train']['main'],
        model_save_path=model_save_path,
        dataname=dataname,
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        device=device
    )

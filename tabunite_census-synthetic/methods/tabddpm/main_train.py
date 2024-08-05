import os

import src
from methods.tabddpm.train import train

def main(args):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'

    config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    raw_config = src.load_config(config_path)

    ''' 
    Modification of configs
    '''
    print('START TRAINING TabDDPM')

    train(
        **raw_config['train']['main'],
        **raw_config['diffusion_params'],
        model_save_path=model_save_path,
        dataname=dataname,
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        device=device
    )

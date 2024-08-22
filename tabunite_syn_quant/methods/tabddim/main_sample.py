import os

import src
from methods.tabddim.sample import sample

def main(args):
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    sample_save_path = args.save_path

    args.train = True

    raw_config = src.load_config(config_path)

    ''' 
    Modification of configs
    '''
    print('START SAMPLING')

    sample(
        **raw_config['train']['main'],
        **raw_config['diffusion_params'],
        model_save_path=model_save_path,
        dataname=dataname,
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        device=device
    )
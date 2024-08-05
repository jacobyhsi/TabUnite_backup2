import os
import argparse

from methods.pskddpm.train import train

import src


def main(args):

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    real_data_path = f'data/{dataname}'

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    args.train = True
    raw_config = src.load_config(config_path)

    ''' 
    Modification of configs
    '''
    print('START TRAINING: pskDDPM')
    
    train(
        **raw_config['train']['main'],
        **raw_config['diffusion_params'],
        model_save_path=model_save_path,
        dataname=dataname,
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        device=device
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--dataname', type = str, default = 'adult')
    parser.add_argument('--gpu', type = int, default=0)

    args = parser.parse_args()
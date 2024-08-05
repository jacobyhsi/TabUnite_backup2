import os

import src
from oheflow.sample import sample

def main(args):
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    real_data_path = f'data/{dataname}'
    sample_save_path = args.save_path

    args.train = True
    
    raw_config = src.load_config(config_path)

    ''' 
    Modification of configs
    '''
    print('START SAMPLING')
    
    sample(
        num_samples=raw_config['sample']['num_samples'],
        batch_size=raw_config['sample']['batch_size'],
        disbalance=raw_config['sample'].get('disbalance', None),
        **raw_config['diffusion_params'],
        model_save_path=model_save_path,
        sample_save_path=sample_save_path,
        real_data_path=real_data_path,
        task_type=raw_config['task_type'],
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        device=device,
        ddim=args.ddim,
        steps=args.steps
    )


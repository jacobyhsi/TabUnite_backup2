import argparse

from methods.dicflow.main_train import main as train_dicflow
from methods.dicflow.main_sample import main as sample_dicflow

from methods.i2bflow.main_train import main as train_i2bflow
from methods.i2bflow.main_sample import main as sample_i2bflow

from methods.tabddpm.main_train import main as train_tabddpm
from methods.tabddpm.main_sample import main as sample_tabddpm

def execute_function(method, mode):
    if method == 'vae':
        mode = 'train'

    main_fn = eval(f'{mode}_{method}')

    return main_fn

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
    parser.add_argument('--method', type=str, default='tabsyn', help='Method: tabsyn or baseline.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')


    # configs for traing TabSyn's VAE
    parser.add_argument('--max_beta', type=float, default=1e-3, help='Maximum beta')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Batch size.')

    # configs for sampling
    parser.add_argument('--save_path', type=str, default=None, help='Path to save synthetic data.')

    args = parser.parse_args()
    return args

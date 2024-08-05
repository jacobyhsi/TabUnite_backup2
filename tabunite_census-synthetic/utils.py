import argparse
from methods.i2bflow.main_train import main as train_i2bflow
from methods.i2bflow.main_sample import main as sample_i2bflow

from methods.dicflow.main_train import main as train_dicflow
from methods.dicflow.main_sample import main as sample_dicflow

from methods.oheflow.main_train import main as train_oheflow
from methods.oheflow.main_sample import main as sample_oheflow

from methods.tabflow.main_train import main as train_tabflow
from methods.tabflow.main_sample import main as sample_tabflow

from methods.i2bddpm.main_train import main as train_i2bddpm
from methods.i2bddpm.main_sample import main as sample_i2bddpm

from methods.dicddpm.main_train import main as train_dicddpm
from methods.dicddpm.main_sample import main as sample_dicddpm

from methods.oheddpm.main_train import main as train_oheddpm
from methods.oheddpm.main_sample import main as sample_oheddpm

from methods.tabddpm.main_train import main as train_tabddpm
from methods.tabddpm.main_sample import main as sample_tabddpm

from methods.pskddpm.main_train import main as train_pskddpm
from methods.pskddpm.main_sample import main as sample_pskddpm

from methods.pskflow.main_train import main as train_pskflow
from methods.pskflow.main_sample import main as sample_pskflow

def execute_function(method, mode):
    if method == 'vae':
        mode = 'train'

    main_fn = eval(f'{mode}_{method}')

    return main_fn

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='beijing', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
    parser.add_argument('--method', type=str, default='i2bflow', help='Method: tabunite-i2bflow/tabunite-dicflow or baseline.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')


    # configs for traing TabSyn's VAE
    parser.add_argument('--max_beta', type=float, default=1e-3, help='Maximum beta')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Batch size.')

    # configs for sampling
    parser.add_argument('--save_path', type=str, default=None, help='Path to save synthetic data.')

    args = parser.parse_args()
    return args

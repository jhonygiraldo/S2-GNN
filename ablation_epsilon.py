import logging

import numpy as np

from s2gnn.utils.tools import load_graph_data, set_device
from s2gnn.utils.parser import parse_args
from s2gnn.datasets.seeds import TEST_SEEDS
from s2gnn.datasets.constants import DATASET_PATHS
from s2gnn.runner import run


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def set_best_hyperparameters(config):
    """
    Sets the best hyperparameters for the specific dataset.

    args:
    - config (dict): Configuration dictionary containing dataset and hyperparameters.

    returns:
    - dict: Updated configuration dictionary with the best hyperparameters for the dataset.
    """
    best_hyperparameters = {
        'cancer_b': {
            'lr': 0.0177,
            'weight_decay': 6e-4,
            'hidden_units': 32,
            'n_layers_set': [2],
            'dropout': 0.3077,
            'aggregation': 'concat',
            'alpha': 6,
        },
        'cancer': {
            'lr': 0.0085,
            'weight_decay': 1e-4,
            'hidden_units': 64,
            'n_layers_set': [3],
            'dropout': 0.3525,
            'aggregation': 'concat',
            'alpha': 6,
        },
        '20news': {
            'lr': 0.0057,
            'weight_decay': 4e-4,
            'hidden_units': 32,
            'n_layers_set': [2],
            'dropout': 0.4569,
            'aggregation': 'linear',
            'alpha': 4,
        },
        'activity': {
            'lr': 0.0057,
            'weight_decay': 4e-4,
            'hidden_units': 64,
            'n_layers_set': [2],
            'dropout': 0.3763,
            'aggregation': 'concat',
            'alpha': 6,
        },
        'isolet': {
            'lr': 0.0071,
            'weight_decay': 6e-4,
            'hidden_units': 64,
            'n_layers_set': [2],
            'dropout': 0.6219,
            'aggregation': 'concat',
            'alpha': 2,
        }
    }
    
    if config['dataset'] in best_hyperparameters:
        config.update(best_hyperparameters[config['dataset']])

    return config


def main(config=None, data_folder='data', output_folder='output'):
    """
    Main function to run the ablation study on epsilon values.

    args:
    - config (dict, optional): Configuration dictionary containing dataset and hyperparameters. Default is None.
    - data_folder (str, optional): Path to the folder containing the data. Default is 'data'.
    - output_folder (str, optional): Path to the folder to save the output. Default is 'output'.

    returns:
    - None
    """
    if config is None:
        return

    set_device(config)
    set_best_hyperparameters(config)

    name_folder, init_name = DATASET_PATHS.get(config["dataset"], [None, None])
    config['output_folder'] = output_folder

    epsilon_ablation = np.arange(0, 2.1, 0.1)
    for epsilon in epsilon_ablation:
        config['epsilon'] = epsilon
        data, n = load_graph_data(
            config, 
            root_folder=data_folder, 
            name_folder=name_folder, 
            init_name=init_name
        )
        run(data, n, config, seeds=TEST_SEEDS)

if __name__ == '__main__':
    config = parse_args(train_mode=True)
    main(config, data_folder='data', output_folder='ablation')
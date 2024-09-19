import os
import sys
import pickle
import logging
from typing import Dict, Any

import numpy as np
from tqdm import tqdm

from s2gnn.utils.parser import parse_args
from s2gnn.utils.tools import (
    get_filename_base, 
    load_graph_data, 
    set_device, 
    set_seed
)
from s2gnn.datasets.seeds import VAL_SEEDS
from s2gnn.datasets.constants import DATASET_PATHS
from s2gnn.runner import run


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# to test reproducibility
set_seed(123)


def initialize_search_spaces(
        config: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
    """
    Initialize the search spaces for hyperparameter optimization.

    args:
    - config (Dict[str, Any]): Configuration dictionary containing dataset and hyperparameters.

    returns:
    - Dict[str, np.ndarray]: Dictionary containing the search spaces for each hyperparameter.
    """
    spaces = {
        'lr_space': np.array([0.005, 0.02]),
        'weight_decay_space': np.array([1e-4, 1e-3]),
        'hidden_units_space': np.array([16, 32, 64]),
        'dropout_space': np.array([0.3, 0.7]),
        'n_layers_space': np.array([2, 3, 4, 5])
    }

    if config["GNN"] in ['SSobGNN', 'SobGNN']:
        spaces.update({
            'alpha_space': np.array([1, 2, 3, 4, 5, 6]),
            'epsilon_space': np.array([0.5, 2]),
            'aggregation_space': np.array(['linear', 'concat'])
        })
    if config["GNN"] in ['GAT', 'Transformer', 'SuperGAT', 'GATv2']:
        spaces['heads_space'] = np.array([1, 2, 3, 4, 5, 6])
    if config["GNN"] == 'Cheby':
        spaces['k_cheby_space'] = np.array([1, 2, 3])
    if config["GNN"] == 'SIGN':
        spaces['k_sign_space'] = np.array([1, 2, 3])

    return spaces


def load_or_initialize_hyperparameters(
        output_folder: str, 
        config: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
    """
    Load existing hyperparameters from a file or initialize new ones.

    args:
    - output_folder (str): Path to the folder where hyperparameters are saved.
    - config (Dict[str, Any]): Configuration dictionary containing dataset and hyperparameters.

    returns:
    - Dict[str, np.ndarray]: Dictionary containing the hyperparameters.
    - np.ndarray: Indices of hyperparameters that are zero.
    - str: Path to the hyperparameters file.
    """
    path_hyperparameters = os.path.join(
        output_folder, 
        f'{config["GNN"]}_hyperTuning', 
        config["graph"]
    )
    hyperparameters_path = os.path.join(
        path_hyperparameters, 
        f'{config["dataset"]}.pkl'
    )

    if os.path.exists(hyperparameters_path):
        with open(hyperparameters_path, 'rb') as f:
            hyperparameters = pickle.load(f)
        indx_zeros = np.where(hyperparameters["lr"] == 0)[0]
    else:
        hyperparameters = {
            "lr": np.zeros((config["iterations"],)),
            "weight_decay": np.zeros((config["iterations"],)),
            "hidden_units": np.zeros((config["iterations"],), dtype=int),
            "dropout": np.zeros((config["iterations"],)),
            "n_layers_set": np.zeros((config["iterations"],), dtype=int)
        }
        if config["GNN"] in ['SSobGNN', 'SobGNN']:
            hyperparameters.update({
                "alpha": np.zeros((config["iterations"],), dtype=int),
                "epsilon": np.zeros((config["iterations"],)),
                "aggregation": np.zeros((config["iterations"],), dtype=object)
            })
        if config["GNN"] in ['GAT', 'Transformer', 'SuperGAT', 'GATv2']:
            hyperparameters["heads_attention"] = np.zeros((config["iterations"],))
        if config["GNN"] == 'Cheby':
            hyperparameters["k_cheby"] = np.zeros((config["iterations"],), dtype=int)
        if config["GNN"] == 'SIGN':
            hyperparameters["k_sign"] = np.zeros((config["iterations"],), dtype=int)
        indx_zeros = np.where(hyperparameters["lr"] == 0)[0]

    return hyperparameters, indx_zeros, hyperparameters_path


def update_hyperparameters(
        hyperparameters: Dict[str, np.ndarray], 
        spaces: Dict[str, np.ndarray], 
        i: int
    ) -> None:
    """
    Update the hyperparameters for a specific iteration.

    args:
    - hyperparameters (Dict[str, np.ndarray]): Dictionary containing the hyperparameters.
    - spaces (Dict[str, np.ndarray]): Dictionary containing the search spaces for each hyperparameter.
    - i (int): The iteration index to update.

    returns:
    - None
    """
    hyperparameters["lr"][i] = np.round(
        np.random.uniform(spaces['lr_space'][0], spaces['lr_space'][1]), 
        decimals=4
    )
    hyperparameters["weight_decay"][i] = np.round(
        np.random.uniform(spaces['weight_decay_space'][0], spaces['weight_decay_space'][1]), 
        decimals=4
    )
    hyperparameters["hidden_units"][i] = np.random.choice(
        spaces['hidden_units_space']
    )
    hyperparameters["dropout"][i] = np.round(
        np.random.uniform(spaces['dropout_space'][0], spaces['dropout_space'][1]), 
        decimals=4
    )
    hyperparameters["n_layers_set"][i] = np.random.choice(spaces['n_layers_space'])

    if config["GNN"] in ['SSobGNN', 'SobGNN']:
        hyperparameters["alpha"][i] = np.random.choice(spaces['alpha_space'])
        hyperparameters["epsilon"][i] = np.round(
            np.random.uniform(spaces['epsilon_space'][0], spaces['epsilon_space'][1]), 
            decimals=4
        )
        hyperparameters["aggregation"][i] = np.random.choice(spaces['aggregation_space'])
    if config["GNN"] in ['GAT', 'Transformer', 'SuperGAT', 'GATv2']:
        hyperparameters["heads_attention"][i] = np.random.choice(spaces['heads_space'])
    if config["GNN"] == 'Cheby':
        hyperparameters["k_cheby"][i] = np.random.choice(spaces['k_cheby_space'])
    if config["GNN"] == 'SIGN':
        hyperparameters["k_sign"][i] = np.random.choice(spaces['k_sign_space'])


def save_hyperparameters(
        hyperparameters: Dict[str, np.ndarray], 
        hyperparameters_path: str
    ) -> None:
    """
    Save the hyperparameters to a file.

    args:
    - hyperparameters (Dict[str, np.ndarray]): Dictionary containing the hyperparameters.
    - hyperparameters_path (str): Path to the file where hyperparameters will be saved.

    returns:
    - None
    """
    os.makedirs(os.path.dirname(hyperparameters_path), exist_ok=True)
    with open(hyperparameters_path, 'wb') as f:
        pickle.dump(hyperparameters, f)


def main(
        config: Dict[str, Any], 
        data_folder: str = 'data', 
        output_folder: str = 'output'
    ) -> None:
    """
    Main function to run the hyperparameter optimization.

    args:
    - config (Dict[str, Any], optional): Configuration dictionary containing dataset and hyperparameters. Default is None.
    - data_folder (str, optional): Path to the folder containing the data. Default is 'data'.
    - output_folder (str, optional): Path to the folder to save the output. Default is 'output'.

    returns:
    - None
    """
    set_device(config)
    name_folder, init_name = DATASET_PATHS.get(config["dataset"], [None, None])

    spaces = initialize_search_spaces(config)
    hyperparameters, indx_zeros, \
    hyperparameters_path = load_or_initialize_hyperparameters(output_folder, config)

    experiments = []
    for i in range(config["iterations"]):
        sys.stdout.flush()
        repeat_flag = False
        if i not in indx_zeros:
            params = {key: value[i] for key, value in hyperparameters.items()}
            params.update({
                "dataset": config["dataset"], 
                "epochs": config["epochs"], 
                "GNN": config["GNN"]
            })
            filename = os.path.join(
                output_folder, 
                f'{config["GNN"]}_hyperTuning', 
                config["graph"], 
                f'{get_filename_base(params["n_layers_set"], params)}.pkl'
            )
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    best_acc_test_vec, _ = pickle.load(f)
                if np.any(best_acc_test_vec == 0):
                    repeat_flag = True
                    logger.info(f'Repeating experiment {i}.')
                else:
                    logger.info(f'Experiment {i} is fine.')
            else:
                repeat_flag = True
                logger.info('Repeating experiment.')

        if (i in indx_zeros) or repeat_flag:
            experiments.append(i)
            update_hyperparameters(hyperparameters, spaces, i)
            save_hyperparameters(hyperparameters, hyperparameters_path)

    for i in tqdm(experiments, desc='Hyperparameter tuning'):
        logger.info('Search iteration: ' + str(i))
        params = {key: value[i] for key, value in hyperparameters.items()}
        config.update(params)
        data, n = load_graph_data(
            config, 
            root_folder=data_folder, 
            name_folder=name_folder, 
            init_name=init_name
        )
        run(data, n, config, seeds=VAL_SEEDS)


if __name__ == '__main__':
    config = parse_args(train_mode=False)
    main(
        config, 
        data_folder='data', 
        output_folder='results_hyper_graph_classification'
    )
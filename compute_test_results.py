import os
import logging
import pickle
from typing import Tuple, Dict, Any

import numpy as np
import seaborn as sns

from s2gnn.utils.tools import load_graph_data, set_device
from s2gnn.utils.parser import parse_args
from s2gnn.datasets.seeds import TEST_SEEDS
from s2gnn.datasets.constants import DATASET_PATHS
from s2gnn.runner import run


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_hyperparameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load hyperparameters from a pickle file.

    args:
    - config (dict): Configuration dictionary containing dataset and hyperparameters.

    returns:
    - dict: Loaded hyperparameters.
    """
    if config["graph_classification"]:
        path = os.path.join(
            'results_hyper_graph_classification',
            f'{config["GNN"]}_hyperTuning',
            f'{config["graph"]}',
            f'{config["dataset"]}.pkl'
        )
    else:
        path = os.path.join(
            'results_hyper_3',
            f'{config["GNN"]}_hyperTuning',
            f'{config["graph"]}',
            f'{config["dataset"]}.pkl'
        )

    with open(path, 'rb') as f:
        return pickle.load(f)


def generate_file_name_base(
        config: Dict[str, Any], 
        hyperparameters: Dict[str, Any],
        i: int
    ) -> str:
    """
    Generate the base filename for saving results based on hyperparameters.

    args:
    - config (dict): Configuration dictionary containing dataset and hyperparameters.
    - hyperparameters (dict): Dictionary containing hyperparameters.
    - i (int): Index of the current hyperparameter set.

    returns:
    - str: The base filename.
    """
    file_name_base = (
        f'{config["dataset"]}_nL_{int(hyperparameters["n_layers_set"][i])}'
        f'_hU_{int(hyperparameters["hidden_units"][i])}_lr_{hyperparameters["lr"][i]}'
        f'_wD_{hyperparameters["weight_decay"][i]}_dr_{hyperparameters["dropout"][i]}'
        f'_mE_{config["epochs"]}'
    )
    if config["GNN"] in ['SSobGNN', 'SobGNN', 'CascadeGCN']:
        file_name_base += (
            f'_alpha_{int(hyperparameters["alpha"][i])}_epsilon_{hyperparameters["epsilon"][i]}'
            f'_aggregation_{hyperparameters["aggregation"][i]}'
        )
    if config["GNN"] in ['GAT', 'Transformer', 'SuperGAT', 'GATv2']:
        file_name_base += f'_heads_attention_{hyperparameters["heads_attention"][i]}'
    if config["GNN"] == 'Cheby':
        file_name_base += f'_KCheby_{int(hyperparameters["k_cheby"][i])}'
    if config["GNN"] == 'SIGN':
        file_name_base += f'_KSIGN_{int(hyperparameters["k_sign"][i])}'

    return file_name_base


def calculate_performance(
        config: Dict[str, Any], 
        hyperparameters: Dict[str, Any],
        search_iterations: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the performance of each hyperparameter combination using bootstrapping.

    args:
    - hyperparameters (dict): Dictionary containing hyperparameters.
    - config (dict): Configuration dictionary containing dataset and hyperparameters.
    - search_iterations (int): Number of hyperparameter sets to evaluate.

    returns:
    - tuple: Mean validation performance, mean test performance, and standard deviation of test performance.
    """
    val_mean = np.zeros((search_iterations,))
    test_mean = np.zeros((search_iterations,))
    test_std = np.zeros((search_iterations,))

    for i in range(search_iterations):
        file_name_base = generate_file_name_base(config, hyperparameters, i)

        if config["graph_classification"]:
            file_name_results = os.path.join(
                'results_hyper_graph_classification',
                f'{config["GNN"]}_hyperTuning',
                f'{config["graph"]}',
                f'{file_name_base}.pkl'
            )
        else:
            file_name_results = os.path.join(
                'results_hyper',
                f'{config["GNN"]}_hyperTuning',
                f'{config["graph"]}',
                f'{file_name_base}.pkl'
            )

        with open(file_name_results, 'rb') as f:
            best_acc_test_vec, best_acc_val_vec = pickle.load(f)

        val_mean[i] = np.mean(
            sns.algorithms.bootstrap(best_acc_val_vec, func=np.mean, n_boot=1000)
        )
        boots_series = sns.algorithms.bootstrap(
            best_acc_test_vec, 
            func=np.mean, 
            n_boot=1000
        )
        test_mean[i] = np.mean(boots_series)
        test_std[i] = np.max(
            np.abs(sns.utils.ci(boots_series, 95) - np.mean(best_acc_test_vec))
        )

    return val_mean, test_mean, test_std


def find_best_hyperparameters(val_mean: np.ndarray) -> int:
    """
    Find the index of the best hyperparameters based on validation performance.

    args:
    - val_mean (np.ndarray): Array of mean validation performances.

    returns:
    - int: Index of the best hyperparameters.
    """
    best_val_indx = np.where(val_mean == np.max(val_mean))

    if best_val_indx[0].shape[0] > 1:
        return best_val_indx[0][0]

    return best_val_indx[0]


def log_hyperparameters(
        config: Dict[str, Any], 
        hyperparameters: Dict[str, Any],
        best_val_indx: int, 
        test_mean: np.ndarray, 
        test_std: np.ndarray
    ) -> None:
    """
    Log the best hyperparameters and their corresponding performance.

    args:
    - config (dict): Configuration dictionary containing dataset and hyperparameters.
    - hyperparameters (dict): Dictionary containing hyperparameters.
    - best_val_indx (int): Index of the best hyperparameters.
    - test_mean (np.ndarray): Array of mean test performances.
    - test_std (np.ndarray): Array of standard deviations of test performances.

    returns:
    - None
    """
    results_log = (
        f'The best result for {config["GNN"]} with val seeds in {config["dataset"]} dataset, '
        f'with graph {config["graph"]} is '
        f'{test_mean[best_val_indx]}+-{test_std[best_val_indx]}'
    )
    logger.info(results_log)

    hyperparameters_log = (
        f'n_layers: {int(hyperparameters["n_layers_set"][best_val_indx])} hiddenUnits: '
        f'{int(hyperparameters["hidden_units"][best_val_indx])} lr: {hyperparameters["lr"][best_val_indx]} '
        f'weightDecay: {hyperparameters["weight_decay"][best_val_indx]} '
        f'dropout: {hyperparameters["dropout"][best_val_indx]}'
    )

    if config["GNN"] in ['SSobGNN', 'SobGNN', 'CascadeGCN']:
        hyperparameters_log += (
            f' alpha: {int(hyperparameters["alpha"][best_val_indx])} epsilon: {hyperparameters["epsilon"][best_val_indx]} '
            f'aggregation: {hyperparameters["aggregation"][best_val_indx[0][0]]}'
        )
        config.update({
            "alpha": int(hyperparameters["alpha"][best_val_indx]),
            "epsilon": hyperparameters["epsilon"][best_val_indx][0],
            "aggregation": hyperparameters["aggregation"][best_val_indx[0][0]],
        })

    if config["GNN"] in ['GAT', 'Transformer', 'SuperGAT', 'GATv2']:
        hyperparameters_log += f' heads: {hyperparameters["heads_attention"][best_val_indx][0]}'
        config["heads_attention"] = hyperparameters["heads_attention"][best_val_indx][0]

    if config["GNN"] == 'Cheby':
        hyperparameters_log += f' k_cheby: {int(hyperparameters["k_cheby"][best_val_indx])}'
        config["k_cheby"] = int(hyperparameters["k_cheby"][best_val_indx])

    if config["GNN"] == 'SIGN':
        hyperparameters_log += f' k_sign: {int(hyperparameters["k_sign"][best_val_indx])}'
        config["k_sign"] = int(hyperparameters["k_sign"][best_val_indx])

    logger.info(hyperparameters_log)


def main(
        config: Dict[str, Any], 
        data_folder: str = 'data', 
        output_folder: str = 'output'
    ) -> None:
    """
    Main function to run the hyperparameter evaluation and logging.

    args:
    - config (dict, optional): Configuration dictionary containing dataset and hyperparameters. Default is None.
    - data_folder (str, optional): Path to the folder containing the data. Default is 'data'.
    - output_folder (str, optional): Path to the folder to save the output. Default is 'output'.

    returns:
    - None
    """

    set_device(config)

    name_folder, init_name = DATASET_PATHS.get(config["dataset"], [None, None])
    data, n = load_graph_data(config, data_folder, name_folder, init_name)

    config['output_folder'] = output_folder

    hyperparameters = load_hyperparameters(config)
    search_iterations = hyperparameters["lr"].shape[0]

    val_mean, test_mean, test_std = calculate_performance(
        config,
        hyperparameters, 
        search_iterations
    )

    best_val_indx = find_best_hyperparameters(val_mean)
    log_hyperparameters(config, hyperparameters, best_val_indx, test_mean, test_std)

    run(data, n, config, seeds=TEST_SEEDS)


if __name__ == '__main__':
    config = parse_args(train_mode=True)
    output_folder = (
        'results_test_graph_classification' 
        if config["graph_classification"]
        else 'results_test_node_classification'
    )
    main(config, data_folder='data', output_folder=output_folder)
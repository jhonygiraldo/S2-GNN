import os
import pickle
import logging
from typing import Dict, Any

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from s2gnn.utils.parser import parse_args


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(config: Dict[str, Any], output_folder = 'output'):
    """
    Main function to run the ablation study and plot the results.

    args:
    - config (dict, optional): Configuration dictionary containing dataset and hyperparameters. Default is None.
    - output_folder (str, optional): Path to the folder to save the output. Default is 'output'.

    returns:
    - None
    """

    config['device'] = setup_device(config)

    dataset_hyperparams = {
        'cancer_b': {
            'lr': 0.0177, 
            'weight_decay': 6e-4, 
            'hidden_units': 32, 
            'n_layers_set': [2], 
            'dropout': 0.3077, 
            'aggregation': 'concat', 
            'alpha': 6
        },
        'cancer': {
            'lr': 0.0085, 
            'weight_decay': 1e-4, 
            'hidden_units': 64, 
            'n_layers_set': [3],
            'dropout': 0.3525, 
            'aggregation': 'concat', 
            'alpha': 6
        },
        '20news': {
            'lr': 0.0057, 
            'weight_decay': 4e-4, 
            'hidden_units': 32, 
            'n_layers_set': [2],
            'dropout': 0.4569, 
            'aggregation': 'linear', 
            'alpha': 4
        },
        'activity': {
            'lr': 0.0057, 
            'weight_decay': 4e-4, 
            'hidden_units': 64, 
            'n_layers_set': [2],
            'dropout': 0.3763, 
            'aggregation': 'concat', 
            'alpha': 6
        },
        'isolet': {
            'lr': 0.0071, 
            'weight_decay': 6e-4, 
            'hidden_units': 64, 
            'n_layers_set': [2],
            'dropout': 0.6219, 
            'aggregation': 'concat', 
            'alpha': 2
        }
    }

    # Set dataset-specific hyperparameters
    if config['dataset'] in dataset_hyperparams:
        config.update(dataset_hyperparams[config['dataset']])

    config['output_folder'] = output_folder

    epsilon_ablation = np.arange(0, 2.1, 0.1)
    mean_accuracy, std_accuracy = run_ablation_study(config, epsilon_ablation)

    plot_results(config, epsilon_ablation, mean_accuracy, std_accuracy)


def setup_device(config: Dict[str, Any]):
    """
    Set up the computation device (CPU or GPU) based on the configuration.

    args:
    - config (dict): Configuration dictionary containing dataset and hyperparameters.

    returns:
    - torch.device: The device to be used for computation.
    """
    if not config["no_cuda"] and torch.cuda.device_count() > config["gpu_number"]:
        logger.info(f'Setting up GPU {config["gpu_number"]}...')
        return torch.device('cuda', config["gpu_number"])
    else:
        logger.info('GPU not available, setting up CPU...')
        return torch.device('cpu')


def run_ablation_study(config: Dict[str, Any], epsilon_ablation: np.ndarray):
    """
    Run the ablation study by varying the epsilon parameter and evaluating the model.

    args:
    - config (dict): Configuration dictionary containing dataset and hyperparameters.
    - epsilon_ablation (np.ndarray): Array of epsilon values to evaluate.

    returns:
    - tuple: Mean accuracy and standard deviation of accuracy for each epsilon value.
    """
    mean_accuracy = np.zeros(epsilon_ablation.shape[0])
    std_accuracy = np.zeros(epsilon_ablation.shape[0])

    for i, epsilon in enumerate(epsilon_ablation):
        config['epsilon'] = epsilon
        file_name_base = create_file_name(config, epsilon)
        logger.info(file_name_base)

        with open(file_name_base, 'rb') as f:
            _, _, _, best_acc_test_vec, _, _, _ = pickle.load(f)

        mean_accuracy[i] = np.mean(best_acc_test_vec[:, -1])
        std_accuracy[i] = np.std(best_acc_test_vec[:, -1])
    
    return mean_accuracy, std_accuracy


def create_file_name(config: Dict[str, Any], epsilon: float):
    """
    Create the file name for saving results based on the configuration and epsilon value.

    args:
    - config (dict): Configuration dictionary containing dataset and hyperparameters.
    - epsilon (float): The epsilon value for the current ablation study iteration.

    returns:
    - str: The generated file name.
    """
    base_name = (
        f"ablation/{config['GNN']}/{config['graph']}/{config['dataset']}_nL_"
        f"{int(config['n_layers_set'][0])}_hU_{config['hidden_units']}_lr_"
        f"{config['lr']}_wD_{config['weight_decay']}_dr_{config['dropout']}_mE_"
        f"{config['epochs']}"
    )

    if config["GNN"] in ['SSobGNN', 'SobGNN', 'CascadeGCN']:
        base_name += (
            f"_alpha_{config['alpha']}_epsilon_{epsilon}_"
            f"aggregation_{config['aggregation']}.pkl"
        )
    
    return base_name


def plot_results(
        config: Dict[str, Any],
        epsilon_ablation: np.ndarray, 
        mean_accuracy: np.ndarray, 
        std_accuracy: np.ndarray):
    """
    Plot the results of the ablation study.

    args:
    - epsilon_ablation (np.ndarray): Array of epsilon values.
    - mean_accuracy (np.ndarray): Array of mean accuracies for each epsilon value.
    - std_accuracy (np.ndarray): Array of standard deviations of accuracies for each epsilon value.
    - config (dict): Configuration dictionary containing dataset and hyperparameters.

    returns:
    - None
    """
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times'
    font_size = 28

    font = {'size': font_size - 2}
    matplotlib.rc('font', **font)

    fig = plt.figure(tight_layout=True, figsize=(7.2, 5.7))

    plt.plot(epsilon_ablation, mean_accuracy, linestyle='-', color='b')

    yerr0 = mean_accuracy - std_accuracy
    yerr1 = mean_accuracy + std_accuracy
    plt.fill_between(epsilon_ablation, yerr0, yerr1, color='b', alpha=0.5)

    plt.xlabel(r'$\epsilon$', fontsize=font_size - 1)
    plt.ylabel(r'Accuracy', fontsize=font_size - 1)

    dataset_titles = {
        "cancer_b": (r"Cancer-B", [0.85, 1]),
        "cancer": (r"Cancer-M", [0.6, 0.75]),
        "20news": (r"20News", [0.6, 0.75]),
        "activity": (r"HAR", [0.85, 1]),
        "isolet": (r"Isolet", [0.65, 0.9])
    }

    if config["dataset"] in dataset_titles:
        title, ylim = dataset_titles[config["dataset"]]
        plt.title(title)
        plt.ylim(ylim)

    file_name_plot = os.path.join("ablation", f"{config['dataset']}.svg")
    fig.savefig(file_name_plot, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    config = parse_args(train_mode=True)
    main(config, output_folder='ablation')
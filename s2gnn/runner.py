import sys
import time
import logging
from argparse import Namespace
from typing import List, Dict, Any

import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
from torch_geometric.data import Data

from s2gnn.utils.learning import train, test
from s2gnn.utils.tools import get_model, save_results, set_seed
from s2gnn.datasets.helpers import set_train_val_test_split


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_data_for_dataset(
        data: Data, 
        config: Dict[str, Any], 
        seed: int, 
        n: int
    ) -> Data:
    """
    Prepare the dataset for training by setting the train, validation, and test splits.

    args:
        data (Data): The dataset.
        config (Dict[str, Any]): The arguments dictionary.
        seed (int): The random seed.
        n (int): The number of samples.

    Returns:
        Data: The dataset with train, validation, and test splits.
    """
    if config["dataset"] in ['Cora', 'Citeseer', 'Pubmed']:
        num_development = 1500
    else:
        num_development = int(0.55 * n)
    return set_train_val_test_split(
                seed, 
                data, 
                dataset_name=config["dataset"], 
                num_development=num_development
            ).to(config["device"])


def initialize_optimizer(
        model: torch.nn.Module, 
        config: Dict[str, Any]
    ) -> torch.optim.Optimizer:
    """
    Initialize the optimizer for the model.

    args:
        model (torch.nn.Module): The model to optimize.
        config (Dict[str, Any]): The arguments dictionary.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if (config["GraphDifussion"] or config["graph_classification"]) \
        and (config["GNN"] not in ["SGC"]):
        return torch.optim.Adam([
            dict(params=model.convs[0].parameters(), weight_decay=config["weight_decay"]),
            {
                'params': list([p for l in model.convs[1:] for p in l.parameters()]), 
                'weight_decay': 0
            }
        ], lr=config["lr"])
    else:
        return torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )


def run_epoch(
        model: torch.nn.Module, 
        data: Data, 
        optimizer: torch.optim.Optimizer,
        epoch: int, 
        config: Dict[str, Any], 
        loss_train_vec: np.ndarray, 
        loss_val_vec: np.ndarray, 
        loss_test_vec: np.ndarray, 
        best_acc_test_vec: np.ndarray, 
        err_train_vec: np.ndarray, 
        err_test_vec: np.ndarray, 
        err_val_vec: np.ndarray, 
        cont_repetition: int, 
        n_layers: int
    ) -> None:
    """
    Run a single training epoch and log the results.

    args:
        model (torch.nn.Module): The model to train.
        data (Data): The dataset.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): The current epoch number.
        config (Dict[str, Any]): The arguments dictionary.
        loss_train_vec (np.ndarray): Array to store training losses.
        loss_val_vec (np.ndarray): Array to store validation losses.
        loss_test_vec (np.ndarray): Array to store test losses.
        best_acc_test_vec (np.ndarray): Array to store best test accuracies.
        err_train_vec (np.ndarray): Array to store training errors.
        err_test_vec (np.ndarray): Array to store test errors.
        err_val_vec (np.ndarray): Array to store validation errors.
        cont_repetition (int): The current repetition count.
        n_layers (int): The number of layers in the model.
    """
    start_time = time.time()
    loss_train = train(model, data, optimizer)
    results = test(model, data)
    train_acc, loss_train = results['train']['acc'], results['train']['loss']
    val_acc, loss_val = results['val']['acc'], results['val']['loss']
    test_acc, loss_test = results['test']['acc'], results['test']['loss']

    loss_train_vec[cont_repetition, epoch] = loss_train
    loss_val_vec[cont_repetition, epoch] = loss_val
    loss_test_vec[cont_repetition, epoch] = loss_test
    err_test_vec[cont_repetition, epoch] = 1 - test_acc
    err_val_vec[cont_repetition, epoch] = 1 - val_acc
    err_train_vec[cont_repetition, epoch] = 1 - train_acc

    if val_acc > best_acc_test_vec[cont_repetition, epoch]:
        best_acc_test_vec[cont_repetition, epoch] = test_acc

    if not config["hyperparam_mode"] and epoch == config["epochs"] - 1:
        end_time = time.time()
        log = (
            'n_layers = {:02d}, n_epochs = {:03d}, learning rate = {:.6f}, '
            'Time = {:.4f} seg'
        )
        logger.info(
            log.format(
                n_layers, 
                epoch + 1, 
                optimizer.param_groups[0]['lr'], 
                end_time - start_time
            )
        )
        log ='Loss train = {:.4f}, Loss val = {:.4f}, Loss test = {:.4f}'
        logger.info(
            log.format(
                loss_train, 
                loss_val, 
                loss_test
            )
        )
        log = (
            'Train acc = {:.4f}, Best val acc = {:.4f}, '
            'Best test acc = {:.4f}, Error test = {:.4f}'
        )
        logger.info(
            log.format(
                train_acc, 
                best_acc_test_vec[cont_repetition, epoch], 
                test_acc, 
                err_test_vec[cont_repetition, epoch], 
            )
        )

    sys.stdout.flush()


def run(
        data: Data, 
        n: int, 
        config: Dict[str, Any], 
        seeds: List[int]
    ) -> None:
    """
    Main function to run the training process over multiple seeds and layers.

    args:
        data (Data): The dataset.
        n (int): The number of samples.
        config (Dict[str, Any]): The arguments dictionary.
        seeds (List[int]): List of seeds for reproducibility.
    """
    logger.info(Namespace(**config))

    iterator = ([config["n_layers_set"]] 
                if config["hyperparam_mode"] 
                else config["n_layers_set"]
            )

    results = {
        'hyper': [],
        'train': []
    }

    for n_layers in iterator:
        loss_train_vec = np.zeros((len(seeds), config["epochs"]))
        loss_val_vec = np.zeros((len(seeds), config["epochs"]))
        loss_test_vec = np.zeros((len(seeds), config["epochs"]))
        best_acc_test_vec = np.zeros((len(seeds), config["epochs"]))
        best_acc_val_vec = np.zeros((len(seeds), config["epochs"]))
        err_train_vec = np.zeros((len(seeds), config["epochs"]))
        err_test_vec = np.zeros((len(seeds), config["epochs"]))
        err_val_vec = np.zeros((len(seeds), config["epochs"]))

        for cont_repetition, seed in enumerate(seeds):
            logger.info(f'Executing repetition {cont_repetition}')
            set_seed(seed)

            data = get_data_for_dataset(data, config, seed, n)
            model = get_model(config, data, n_layers).to(config["device"])
            optimizer = initialize_optimizer(model, config)

            for epoch in tqdm(range(config["epochs"]), desc='Epoch: '):
                run_epoch(
                    model, 
                    data,
                    optimizer, 
                    epoch, 
                    config, 
                    loss_train_vec, 
                    loss_val_vec, 
                    loss_test_vec, 
                    best_acc_test_vec, 
                    err_train_vec, 
                    err_test_vec, 
                    err_val_vec, 
                    cont_repetition, 
                    n_layers
                )

        results['hyper'].append([
            best_acc_test_vec[:, -1], 
            best_acc_val_vec[:, -1]
        ])
        results['train'].append([
            loss_train_vec, 
            loss_val_vec, 
            loss_test_vec, 
            best_acc_test_vec, 
            err_test_vec, 
            err_val_vec, 
            err_train_vec
            ])

        save_results(config, results, n_layers)

        if config.get("verbose", False):
            acc_test_vec_test = best_acc_test_vec[:, -1]
            boots_series = sns.algorithms.bootstrap(
                acc_test_vec_test, 
                func=np.mean, 
                n_boot=1000
            )
            test_std_test_seeds = np.max(
                np.abs(sns.utils.ci(boots_series, 95) - np.mean(acc_test_vec_test))
            )
            results_log = (
                f'The result for S-SobGNN method in {config["dataset"]} dataset is '
                f'{np.mean(boots_series)} +- {test_std_test_seeds}'
            )
            logger.info(results_log)
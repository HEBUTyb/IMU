
import warnings
warnings.filterwarnings('ignore')

import torch
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds
import numpy as np
import os
import time
from src.dataset import EUROCDataset, load_initial_conditions
from src.losses import INSIntegrationLoss

def main():
    # Data paths
    data_dir = r'F:\数据'
    predata_dir = r'F:\数据'
    mode = 'train'
    address = r'E:\result\euroc\2023_08_18_14_09_47'

    # Network parameters
    net_class = sn.GyroNet
    net_params = {
        'in_dim': 6,
        'out_dim': 6,
        'c0': 16,
        'dropout': 0.1,
        'ks': [7, 7, 7, 7],
        'ds': [4, 4, 4],
        'momentum': 0.1,
        'gyro_std': [1 * np.pi / 180, 2 * np.pi / 180, 5 * np.pi / 180],
    }

    # Dataset parameters
    dataset_params = {
        'data_dir': data_dir,
        'predata_dir': predata_dir,
        'train_seqs': ['MH_01_easy'],
        'val_seqs': ['MH_01_easy'],
        'test_seqs': ['MH_02_easy'],
        'N': 32 * 5,
        'min_train_freq': 16,
        'max_train_freq': 32,
        'dt': 0.005,
    }

    # Initialize dataset
    dataset = EUROCDataset(mode=mode, **dataset_params)

    # Load initial conditions
    initial_conditions = load_initial_conditions(dataset, sequence_index=0)

    # Training parameters
    train_params = {
        'optimizer_class': torch.optim.Adam,
        'optimizer': {'lr': 0.01, 'weight_decay': 1e-4, 'amsgrad': False},
        'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'scheduler': {'T_0': 10, 'T_mult': 1, 'eta_min': 1e-6},
        'dataloader': {
            'batch_size': 1,
            'shuffle': True,
            'num_workers': 4,
            'pin_memory': True
        },
        'freq_val': 5,
        'n_epochs': 100,
        'loss_class': INSIntegrationLoss,  # 确保这里是'loss_class'
        'loss_params': {  # 以及这里是'loss_params'
            'initial_conditions': initial_conditions,
            'dt': dataset_params['dt']
        }
    }

    if mode == 'train':
        start_time = time.time()
        # Update the instantiation with required arguments
        learning_process = lr.GyroLearningBasedProcessing(
            res_dir=address, #train_params['res_dir'],
            tb_dir=f"{address}_log", #train_params['tb_dir'],
            net_class=net_class,
            net_params=net_params,
            address=address,
            dt=dataset_params['dt'],
            train_params=train_params,  # Include train_params
            dataset_class=EUROCDataset,  # Include dataset_class
            dataset_params=dataset_params  # Include dataset_params
        )

        # Start the training process
        learning_process.train(dataset_class=EUROCDataset, dataset_params=dataset_params, train_params=train_params)
        end_time = time.time()
        print(f'Training finished! Spend: {(end_time - start_time) / 60.0} mins')

if __name__ == '__main__':
    main()

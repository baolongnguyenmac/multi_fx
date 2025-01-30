from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import DistributionLoss, MSE

import os
import json
import pandas as pd
import json
import itertools
import argparse

from data.dataloader import DataLoader
from common.constants import *
from common.common import compute_metrics

def add_log(log:dict, count:int, input_size:int, n_stacks:int, kernel_size:int, mlp_units:int, lr:float, freq:list[int], interpolation_mode:str, metric_log:dict):
    # init
    log[count] = {}

    # param
    log[count]["input_size"] = input_size
    log[count]["n_stacks"] = n_stacks
    log[count]["kernel_size"] = kernel_size
    log[count]["mlp_units"] = mlp_units
    log[count]["lr"] = lr
    log[count]["freq"] = freq
    log[count]["interpolation_mode"] = interpolation_mode

    # metrics
    for key in metric_log:
        log[count][key] = metric_log[key]

def write_log(log:dict, dataset_dir:str, label:int):
    with open(os.path.join(dataset_dir, f'{label}.json'), 'w') as fo:
        json.dump(log, fo)

def run(df:pd.DataFrame, param_dict:dict, columns:list[str]):
    loss_fn = param_dict['loss_fn']
    model = NHITS(
        # const
        h=1, loss=loss_fn, valid_loss=loss_fn, max_steps=500, hist_exog_list=columns, val_check_steps=5,
        # early_stop_patience_steps=5,

        # finetune
        n_pool_kernel_size = param_dict['n_pool_kernel_size'],
        n_freq_downsample = param_dict['n_freq_downsample'],
        learning_rate = param_dict['learning_rate'],
        input_size = param_dict['input_size'],

        # semi-fixed: minor modification can be done on these params to change the structure of network
        stack_types = 3 * ["identity"],
        n_blocks = 3 * [1],
        mlp_units = 3 * [[512, 512]],
        interpolation_mode = 'linear',
        batch_size = 256,
    )

    nf = NeuralForecast(models=[model], freq='h')
    rs_frame = nf.cross_validation(df=df, n_windows=None, val_size=len(df)//5, test_size=len(df)//5)
    return rs_frame

if __name__ == '__main__':
    # get dataset from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, required=True)
    input_dict = vars(parser.parse_args())

    # config dataset and model
    dataset = input_dict['dataset']
    mode = CLF
    dataloader = DataLoader(dataset=dataset)

    # create baseline_dir if needed
    baseline_dir = os.path.join(PRETRAINED_DIR, 'baseline')
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)

    # create a folder for each dataset in baseline_dir
    dataset_dir = os.path.join(baseline_dir, dataset)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # hyper-param search space
    pooling_sizes = [[8,4,1]]
    freqs = [[168,24,1]]
    lrs = [0.001]
    input_sizes = [30]
    loss_fn = DistributionLoss('Bernoulli') if mode==CLF else MSE() if mode==REG else None

    # predict all columns (1 by 1) in the dataset
    # since NHITS doesn't take much time to run
    # i decided not to split the fine-tune process into jobs
    # 75 models will be run sequentially
    for idx, label in enumerate(dataloader.columns):
        # get data and the given label
        print(f'\nPredict on label {idx+1}/{len(dataloader.columns)}\n')
        df, mean_, std_ = dataloader.get_data(label)

        # fine-tune
        count = 0
        log = {}
        hyper_param_combinations = list(itertools.product(pooling_sizes, freqs, lrs, input_sizes))
        for pooling_size, freq, lr, input_size in hyper_param_combinations:
            param_dict = {
                "n_pool_kernel_size": pooling_size,
                "n_freq_downsample": freq,
                "learning_rate": lr,
                "input_size": input_size,
                "loss_fn": DistributionLoss('Bernoulli') if mode==CLF else MSE() if mode==REG else None
            }
            rs_frame = run(df, param_dict, dataloader.columns)

            # compute metrics
            if mode == CLF:
                rs_frame['NHITS'] = (rs_frame['NHITS'] > 0.5) * 1
                acc, precision, recall, f1 = compute_metrics(rs_frame['NHITS'], rs_frame['y'])
                metric_log = {
                    "acc": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            elif mode == REG:
                mse = compute_metrics(rs_frame['NHITS'], rs_frame['y']) * (std_**2)
                metric_log = {'mse': mse}

            # print out the metric
            count += 1
            print(f'\nModel {count+1}/{len(hyper_param_combinations)}')
            print('=============================')
            for key in metric_log:
                print(f'{key}:\t\t{metric_log[key]:.5f}')
            print('=============================\n')

            add_log(log, count, input_size, 3, pooling_size, 512, lr, freq, 'linear', metric_log)

        # save log for 75 models for a predicted label
        write_log(log, dataset_dir, label)

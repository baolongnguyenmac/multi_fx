from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import DistributionLoss

import os
import json
import pandas as pd
import json

from data.pre_process import get_data_for_nhits
from common.constants import *
from common.common import compute_metrics

def add_log(log:dict, count:int, input_size:int, n_stacks:int, kernel_size:int, mlp_units:int, lr:float, freq:list[int], interpolation_mode:str, acc:float, precision:float, recall:float, f1:float):
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
    log[count]["acc"] = acc
    log[count]["precision"] = precision
    log[count]["recall"] = recall
    log[count]["f1"] = f1

def write_log(log:dict, file_name:str):
    with open(os.path.join(PRETRAINED_DIR, 'baseline', file_name), 'w') as fo:
        json.dump(log, fo)

def run(dataset:str, label:int, columns:list[str]):
    data_dir = os.path.join(RAW_DATA_DIR, dataset, 'all')
    data_dict = get_data_for_nhits(data_dir=data_dir, label=label)

    pooling_sizes = [[2,2,2], [4,4,4], [8,8,8], [8,4,1], [16,8,1]]
    freqs = [[168,24,1], [24,12,1], [180,60,1], [40,20,1], [64,8,1]]
    lrs = [0.001]
    input_sizes = [5,20,30]

    count = 0
    num_models = len(pooling_sizes)*len(freqs)*len(lrs)*len(input_sizes)
    log = {}
    for pooling_size in pooling_sizes:
        for freq in freqs:
            for lr in lrs:
                for input_size in input_sizes:
                    model = NHITS(
                        # const
                        h=1, loss=DistributionLoss('Bernoulli'), valid_loss=DistributionLoss('Bernoulli'), max_steps=500, hist_exog_list=columns, val_check_steps=5,
                        # early_stop_patience_steps=5,

                        # finetune
                        input_size = input_size,
                        stack_types = 3*["identity"],
                        n_blocks = 3 * [1],
                        n_freq_downsample = freq,
                        n_pool_kernel_size = pooling_size,
                        mlp_units = 3 * [[512, 512]],
                        learning_rate = lr,
                        interpolation_mode = 'linear',
                        batch_size=256
                    )

                    # fit model
                    nf = NeuralForecast(models=[model], freq='h')
                    y_hat = nf.cross_validation(df=data_dict, n_windows=None, val_size=len(data_dict)//5, test_size=len(data_dict)//5)
                    y_hat['NHITS'] = (y_hat['NHITS'] > 0.5) * 1

                    # compute acc
                    acc, precision, recall, f1 = compute_metrics(y_hat['NHITS'], y_hat['y'])
                    print(f'\nModel {count+1}/{num_models}')
                    print('=============================')
                    print(f'Acc:\t\t{acc:.5f}')
                    print(f'Precision:\t{precision:.5f}')
                    print(f'Recall:\t\t{recall:.5f}')
                    print(f'F1:\t\t{f1:.5f}')
                    print('=============================\n')
                    count += 1

                    add_log(log, count, input_size, 3, pooling_size, 512, lr, freq, 'linear', acc, precision, recall, f1)
    write_log(log=log, file_name=f'NHITS-{dataset}-{label}.json')

if __name__ == '__main__':
    # config data right here
    dataset = WTH
    data_dir = os.path.join(RAW_DATA_DIR, dataset)

    # extract columns
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            tmp_df = pd.read_csv(os.path.join(data_dir, filename))
            columns = list(tmp_df.columns)[1:]
            break

    # run stuff
    labels = columns
    summary = {}
    for idx, label in enumerate(labels):
        print(f'\nPredict on label {idx+1}/{len(labels)}\n')
        run(dataset, idx, columns)

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import DistributionLoss

import os
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

def write_log(log:dict, file_name:str='NHITS_USD-JPY.json'):
    with open(os.path.join(PRETRAINED_DIR, 'baseline', file_name), 'w') as fo:
        json.dump(log, fo)

def run():
    data_ = get_data_for_nhits('/home/s2210434/fx/fxdata')

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
                        h=1, loss=DistributionLoss('Bernoulli'), valid_loss=DistributionLoss('Bernoulli'), max_steps=500, hist_exog_list = ['open', 'high', 'low', 'close'], val_check_steps=5,
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
                    y_hat = nf.cross_validation(df=data_, n_windows=None, val_size=len(data_)//5, test_size=len(data_)//5)
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
    write_log(log)

if __name__ == '__main__':
    # data_ = prepare_data()
    # fine_tune(data_)
    run()



























































# def prepare_data(data_dir:str='./data'):
#     data_dict = _get_data(1, batch_size=-1, data_dir=data_dir, test_size=0.4)
#     # bỏ đi mẫu đầu của X và mẫu cuối của y để mô phòng chính xác time-series dataset
#     # bởi vì `get_data` trả về 1 df đã construct để train bình thường còn NeuralForecast yêu cầu cái df time-series như ban đầu
#     X = data_dict[0]['support_X'].reshape(-1, 4)[1:]
#     y = data_dict[0]['support_y'].reshape(-1,)[:-1]

#     X = X.astype(np.float32)
#     y = y.astype(np.float32)

#     data_ = pd.DataFrame(X, columns=['open', 'high', 'low', 'close'])
#     data_['y'] = y
#     data_['unique_id'] = 0
#     data_['ds'] = pd.date_range(start='2000-01-01', periods=len(data_), freq='D')
#     return data_[['unique_id', 'ds', 'open', 'high', 'low', 'close', 'y']]

# def get_accuracy(y, y_hat):
#     return np.mean(y==y_hat).item()

# def add_log(log:dict, count:int, input_size:int, n_stacks:int, kernel_size:int, mlp_units:int, lr:float, freq:list[int], interpolation_mode:str, acc:float):
#     log[count] = {}
#     log[count]["input_size"] = input_size
#     log[count]["n_stacks"] = n_stacks
#     log[count]["kernel_size"] = kernel_size
#     log[count]["mlp_units"] = mlp_units
#     log[count]["lr"] = lr
#     log[count]["freq"] = freq
#     log[count]["interpolation_mode"] = interpolation_mode
#     log[count]["acc"] = acc

# def write_log(log:dict, file_name:str='log_baseline.json'):
#     with open(os.path.join(LOG_DIR, file_name), 'w') as fo:
#         json.dump(log, fo)

# def fine_tune(data_:pd.DataFrame):
#     set_input_size = [20, 30, 40, 50]
#     set_kernel_size = [2,3,4]
#     set_mlp_units = [32, 64, 128]
#     set_lr = [0.001, 0.0015, 0.005, 0.0055, 0.01]
#     set_freq = [[8, 4, 2, 1, 1], [1, 1, 1, 1, 1], [5, 4, 3, 2, 1]]
#     n_stacks = 5
#     set_epoch = [50, 100, 150, 200, 250] # 60 epochs -> 62% accuracy on test set, 100 epochs -> 99% accuracy on test set

#     log = {}
#     count = 0
#     for input_size in set_input_size:
#         for kernel_size in set_kernel_size:
#             for mlp_units in set_mlp_units:
#                 for lr in set_lr:
#                     for interpolation_mode in ["linear", "nearest", "cubic"]:
#                         for freq in set_freq:
#                             for epochs in set_epoch:
#                                 # init model
#                                 model = NHITS(
#                                     # const
#                                     h=1, loss=DistributionLoss('Bernoulli'), valid_loss=DistributionLoss('Bernoulli'), max_steps=epochs, hist_exog_list = ['open', 'high', 'low', 'close'], val_check_steps=5,
#                                     # early_stop_patience_steps=5,

#                                     # finetune
#                                     input_size = input_size,
#                                     stack_types = n_stacks*["identity"],
#                                     n_blocks = n_stacks * [1],
#                                     n_freq_downsample = freq,
#                                     n_pool_kernel_size = n_stacks * [kernel_size],
#                                     mlp_units = n_stacks * [[mlp_units, mlp_units]],
#                                     learning_rate = lr,
#                                     interpolation_mode = interpolation_mode
#                                 )

#                                 # fit model
#                                 nf = NeuralForecast(models=[model], freq='D')
#                                 y_hat = nf.cross_validation(df=data_, n_windows=None, val_size=len(data_)//5, test_size=len(data_)//5)
#                                 y_hat['NHITS'] = (y_hat['NHITS'] > 0.5) * 1

#                                 # compute acc
#                                 nhits_acc = get_accuracy(y=y_hat['y'], y_hat=y_hat['NHITS'])
#                                 print(f'\n========= NHITS Accuracy: {nhits_acc:.5f} =========\n')

#                                 # log
#                                 add_log(log, count, input_size, n_stacks, kernel_size, mlp_units, lr, freq, interpolation_mode, nhits_acc)
#                                 count += 1

#     # write log
#     write_log(log)

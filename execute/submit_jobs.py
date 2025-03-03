'''
this file should be run from the root directory of the project
>> python -m execute.submit_jobs

it will generate `tmp.sh` to submit the job to server
all `tmp.sh` file assume that they are called from the root directory
'''

import subprocess
import time
import itertools

from common.constants import *

def generate_script(device:str, name_job:str, command:str):
    log_file_path = os.path.join(LOG_DIR, f"{name_job}.log")
    command = f'{command} &>> {log_file_path}'
    name_job = f'{name_job}_job'

    if device == 'cpu':
        class_name = 'DEFAULT'
        resources = 'select=1:ncpus=64'
        timeout = 'walltime=72:00:00'
        load_modules = ''
        test_device = f'echo "Using $(nproc) CPU cores" &>> {log_file_path}'
    elif device == 'gpu':
        class_name = 'GPU-1A'
        resources = 'select=1:ngpus=1'
        timeout = 'walltime=72:00:00'
        load_modules = 'module load cuda'
        test_device = f'nvidia-smi &>> {log_file_path}'
    else:
        raise ValueError(f'device={device} undefined')

    tmp_file = [
        '#!/bin/bash',
        f'#PBS -N {name_job}',
        f'#PBS -q {class_name}',
        f'#PBS -l {resources}',
        f'#PBS -l {timeout}',
        'module purge',
        load_modules,
        'source ~/.bashrc',
        'cd $PBS_O_WORKDIR',

        '\n# Commands to run your job',

        f'echo "Starting {name_job} at `date`" &> {log_file_path}',
        f'conda activate fx_env',
        f'conda env list &>> {log_file_path}',
        test_device,
        command
    ]

    with open(os.path.join(EXE_DIR, 'tmp.sh'), "w") as fout:
        fout.write("\n".join(tmp_file) + "\n")

'''
function finetune_meta_model tunes all of meta-model (LSTM, LSTM+CNN) on all dataset
    [-] multi-fx
        - inner: [0.001, 0.005, 0.01, 0.05]
        - outer: [0.001, 0.0015, 0.005, 0.0055]
    [-] USD/JPY
        - inner: [0.001, 0.005, 0.01, 0.05]
        - outer: [0.005, 0.0055, 0.001, 0.0015]
    [-] ETT
        - inner: [0.001, 0.005, 0.01, 0.05]
        - outer: [0.001, 0.0015, 0.005, 0.0055]
    [-] WTH
        - inner: [0.001, 0.005, 0.01, 0.05]
        - outer: [0.001, 0.0015, 0.005, 0.0055]
    [x] ECL
'''
def finetune_meta_model():
    datasets = [MULTI_FX, USD_JPY, ETT, WTH]
    windows = [20, 30]
    models = [ATT]
    inner_lrs = [0.001, 0.0025, 0.005]
    outer_lrs = [0.005, 0.0055, 0.006, 0.0065]
    old_limit = 78
    limit = 100

    for idx, (dataset, window, model, inner_lr, outer_lr) in enumerate(itertools.product(datasets, windows, models, inner_lrs, outer_lrs)):
        if idx < old_limit:
            continue
        if idx < limit:
            device = 'cpu'
            ncores_limit = 64 if device=='cpu' else 26 if device=='gpu' else 0
            command = f'python -m execute.main -dataset {dataset} -window {window} -model {model} -mode {CLF} -out_shape {2} -inner_lr {inner_lr} -outer_lr {outer_lr} -bt_size {5} -rounds {100} -epochs {9} -ncpus {ncores_limit}'

            generate_script(device=device, name_job=f'{idx}.{dataset}_{model}', command=command)
            subprocess.run(["qsub", os.path.join(EXE_DIR, 'tmp.sh')])
            time.sleep(1)

# once again, since NHITS doesn't take much time to run
# i decided not to split the fine-tune process into jobs
# 75 models will be run sequentially
def finetune_nhits():
    for idx, dataset in enumerate([USD_JPY, ETT, WTH]):
        generate_script(device='cpu', name_job=f'{idx}.{dataset}', command=f'python -m model.baseline_models -dataset {dataset}')
        subprocess.run(["qsub", os.path.join(EXE_DIR, 'tmp.sh')])
        time.sleep(1)

if __name__ == '__main__':
    finetune_meta_model()
    # finetune_nhits()

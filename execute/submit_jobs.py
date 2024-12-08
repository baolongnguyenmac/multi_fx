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

def generate_script(device:str, name_job:str, look_back:int, model_name:str, mode:str, out_shape:int, inner_lr:float, outer_lr:float, bt_size:int, rounds:int, epochs:int):
    ncores_limit = 64 if device=='cpu' else 26 if device=='gpu' else 0
    log_file_path = os.path.join(LOG_DIR, f"{name_job}.log")
    command = f'python -m execute.main -window {look_back} -model {model_name} -mode {mode} -out_shape {out_shape} -inner_lr {inner_lr} -outer_lr {outer_lr} -bt_size {bt_size} -rounds {rounds} -epochs {epochs} -ncpus {ncores_limit} &>> {log_file_path}'
    name_job = f'{name_job}_job'

    if device == 'cpu':
        class_name = 'DEFAULT'
        resources = 'select=1:ncpus=64'
        timeout = 'walltime=24:00:00'
        load_modules = ''
        test_device = f'echo "Using $(nproc) CPU cores" &>> {log_file_path}'
    elif device == 'gpu':
        class_name = 'GPU-1A'
        resources = 'select=1:ngpus=1'
        timeout = 'walltime=24:00:00'
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

        f'echo "Starting {model_name} at `date`" &> {log_file_path}',
        f'conda activate fx_env',
        f'conda env list &>> {log_file_path}',
        test_device,
        command
    ]

    with open(os.path.join(EXE_DIR, 'tmp.sh'), "w") as fout:
        fout.write("\n".join(tmp_file) + "\n")

'''
function finetune_3 fine-tunes all of meta-model (LSTM, LSTM+CNN) on all dataset
    [v] multi-fx
        - inner: [0.001, 0.005, 0.01, 0.05]
        - outer: [0.001, 0.0015, 0.005, 0.0055]
    [v] USD/JPY
        - inner: [0.001, 0.005, 0.01, 0.05]
        - outer: [0.001, 0.0015, 0.005, 0.0055]
    [-] ETT  ---> need to run them again (after analyzing)
        - inner: [0.001, 0.005, 0.01, 0.05]
        - outer: [0.001, 0.0015, 0.005, 0.0055]
    [-] WTH
        - inner: [0.001, 0.005, 0.01, 0.05]
        - outer: [0.001, 0.0015, 0.005, 0.0055]
    [x] ECL
'''
def finetune_3():
    windows = [20, 30]
    models = [LSTM, LSTM_CNN]
    inner_lrs = [0.001, 0.005, 0.01, 0.05]
    outer_lrs = [0.001, 0.0015, 0.005, 0.0055]
    old_limit = 40
    limit = 64

    for idx, (window, model, inner_lr, outer_lr) in enumerate(itertools.product(windows, models, inner_lrs, outer_lrs)):
        if idx < old_limit:
            continue
        if idx < limit:
            generate_script('cpu', f'{idx}.{model}', window, model, CLF, 2, inner_lr, outer_lr, 5, 100, 3)
            subprocess.run(["qsub", os.path.join(EXE_DIR, 'tmp.sh')])
            time.sleep(1)

if __name__ == '__main__':
    finetune_3()

'''
this file should be run from the root directory of the project
>> python -m execute.submit_jobs

it will generate `tmp.sh` to submit the job to server
all `tmp.sh` files are supposed that they are called from the root directory
'''

import subprocess
import time
import itertools

from common.constants import *

def generate_script(device:str, name_job:str, look_back:int, model_name:str, mode:str, inner_lr:float, outer_lr:float, bt_size:int, rounds:int, epochs:int):
    ncores_limit = 64 if device=='cpu' else 26 if device=='gpu' else 0
    log_file_path = os.path.join(LOG_DIR, f"{name_job}.log")
    command = f'python -m execute.main -window {look_back} -model {model_name} -mode {mode} -inner_lr {inner_lr} -outer_lr {outer_lr} -bt_size {bt_size} -rounds {rounds} -epochs {epochs} -ncpus {ncores_limit} &>> {log_file_path}'
    name_job = f'{name_job}_job'

    if device == 'cpu':
        class_name = 'DEFAULT'
        resources = f'select=1:ncpus={ncores_limit}'
        load_modules = ''
        test_device = f'echo "Using $(nproc) CPU cores" &>> {log_file_path}'
    elif device == 'gpu':
        class_name = 'GPU-1'
        resources = 'select=1:ngpus=1'
        load_modules = 'module load cuda'
        test_device = f'nvidia-smi &>> {log_file_path}'
    else:
        raise ValueError(f'device={device} undefined')

    tmp_file = [
        '#!/bin/bash',
        f'#PBS -N {name_job}',
        f'#PBS -q {class_name}',
        f'#PBS -l {resources}',
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

# finetune meta model (based model: CNN, LSTM)
def finetune_1():
    windows = [10, 20, 30]
    models = [LSTM, CNN]
    inner_lrs = [0.001, 0.005, 0.01, 0.05]
    outer_lrs = [0.001, 0.005, 0.0015, 0.0055]

    count = 0
    for window, model, inner_lr, outer_lr in itertools.product(windows, models, inner_lrs, outer_lrs):
        if count < 20:
            count += 1
            continue

        generate_script('cpu', f'{model}_{count}', window, model, CLF, inner_lr, outer_lr, 5, 100, 3)
        subprocess.run(["qsub", os.path.join(EXE_DIR, 'tmp.sh')])
        time.sleep(1)
        count += 1

    # # demo
    # generate_script('gpu', WEAK_LEARNERS[0], 'boosting')
    # subprocess.run(["qsub", os.path.join(EXE_DIR, 'tmp.sh')])

# finetune meta model (based model: LSTM, LSTM+CNN)
def finetune_2():
    windows = [20, 30]
    models = [LSTM, LSTM_CNN]
    inner_lrs = [0.001, 0.005, 0.01, 0.05]
    outer_lrs = [0.001, 0.0015, 0.005, 0.0055]
    # inner_lrs = [0.005, 0.001, 0.01]
    # outer_lrs = [0.0055, 0.005]

    count = 0
    for window, model, inner_lr, outer_lr in itertools.product(windows, models, inner_lrs, outer_lrs):
        generate_script('cpu', f'{model}_{count}', window, model, CLF, inner_lr, outer_lr, 5, 100, 3)
        subprocess.run(["qsub", os.path.join(EXE_DIR, 'tmp.sh')])
        time.sleep(1)
        count += 1

'''
function finetune_3 fine-tunes all of meta-model (LSTM, LSTM+CNN) on all dataset
    [v] multi-fx
        - inner: [0.001, 0.005]
        - outer: [0.005, 0.0055]
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

    count = 0
    for window, model, inner_lr, outer_lr in itertools.product(windows, models, inner_lrs, outer_lrs):
        generate_script('cpu', f'{count}.{model}', window, model, CLF, inner_lr, outer_lr, 5, 100, 3)
        subprocess.run(["qsub", os.path.join(EXE_DIR, 'tmp.sh')])
        time.sleep(1)
        count += 1

if __name__ == '__main__':
    finetune_3()

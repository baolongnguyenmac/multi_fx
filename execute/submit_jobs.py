'''
this file should be run from the root directory of the project
>> python -m execute.submit_jobs

it will generate `tmp.sh` to submit the job to server
all `tmp.sh` file assume that they are called from the root directory
'''

import subprocess
import time

from common.constants import *

def generate_script(device:str, name_job:str, look_back:int, model_name:str, mode:str, inner_lr:float, outer_lr:float, bt_size:int, rounds:int, epochs:int):
    ncores_limit = 64 if device=='cpu' else 26 if device=='gpu' else 0
    log_file_path = os.path.join(LOG_DIR, f"{name_job}.log")
    command = f'python -m execute.main -window {look_back} -model {model_name} -mode {mode} -inner_lr {inner_lr} -outer_lr {outer_lr} -bt_size {bt_size} -rounds {rounds} -epochs {epochs} -ncpus {ncores_limit} &>> {log_file_path}'
    name_job = f'{name_job}_job'

    if device == 'cpu':
        class_name = 'DEFAULT'
        resources = 'select=1:ncpus=64'
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

def submit_based_model():
    current_count = None
    prev_count = 96
    count = 0
    for window in [10, 20, 30]:
        for model in [LSTM, CNN]:
            for inner_lr in [0.001, 0.005, 0.01, 0.05]:
                for outer_lr in [0.001, 0.005, 0.0015, 0.0055]:
                    if count < 20:
                        count += 1
                        continue

                    generate_script('cpu', f'{model}_{count}', window, model, CLF, inner_lr, outer_lr, 5, 100, 3)
                    subprocess.run(["qsub", os.path.join(EXE_DIR, 'tmp.sh')])
                    time.sleep(1)
                    count += 1

                    if count == current_count:
                        return

    # for project_name in [LSTM, CNN]:
    #     generate_script('cpu', project_name)
    #     subprocess.run(["qsub", os.path.join(EXE_DIR, 'tmp.sh')])
    #     time.sleep(3)

    # # demo
    # generate_script('gpu', WEAK_LEARNERS[0], 'boosting')
    # subprocess.run(["qsub", os.path.join(EXE_DIR, 'tmp.sh')])

def submit_tunned_model():
    count = 0
    for window in [20, 30]:
        for model in [LSTM]:
            for inner_lr in [0.005]:
                for outer_lr in [0.0055, 0.005]:
                    generate_script('cpu', f'{model}_{count}', window, model, CLF, inner_lr, outer_lr, 5, 100, 3)
                    subprocess.run(["qsub", os.path.join(EXE_DIR, 'tmp.sh')])
                    time.sleep(1)
                    count += 1

if __name__ == '__main__':
    # submit_based_model()
    submit_tunned_model()

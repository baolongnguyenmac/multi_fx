import tensorflow as tf
import random
random.seed(84)
import argparse
import pandas as pd
import json

from model.meta_model import MAML
from common.constants import *
from data.pre_process import get_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-window', '--look_back_window', type=int, required=True)
    parser.add_argument('-model', '--based_model', type=str, choices=[LSTM, CNN, LSTM_CNN, ATT], default=LSTM, required=False)
    parser.add_argument('-mode', '--mode', type=str, choices=[REG, CLF], default=CLF, required=True)
    parser.add_argument('-inner_lr', '--inner_learning_rate', type=float, required=True) # we need this param in test phase
    parser.add_argument('-outer_lr', '--outer_learning_rate', type=float, default=0.001, required=False)
    parser.add_argument('-bt_size', '--batch_task_size', type=int, default=5, required=False)
    parser.add_argument('-rounds', '--num_rounds', type=int, default=100, required=False)
    parser.add_argument('-epochs', '--num_epochs', type=int, default=3, required=True) # we need this param in test phase
    parser.add_argument('-ncpus', '--num_cpus', type=int, default=8)

    return vars(parser.parse_args())

def run(dataset:str, input_dict:dict, label:int):
    # prep data
    data_dir = os.path.join(RAW_DATA_DIR, dataset)
    data_dict = get_data(look_back=input_dict['look_back_window'], data_dir=data_dir, label=label)

    if 'multi' in dataset:
        # only apply for multi_fx
        # choose randomly 50% clients for training, 25% for validating and 25% for testing
        list_id = set(data_dict.keys())
        list_id_train = random.sample(sorted(list_id), 30)
        list_id_val = random.sample(sorted(list_id.difference(set(list_id_train))), 15)
        list_id_test = list(list_id.difference(set(list_id_train).union(list_id_val)))
    else:
        # bởi vì các dataset khác không multi
        # nên phải đảm bảo tính thứ tự trong huấn luyện
        list_id = list(data_dict.keys())
        list_id_train = list_id[:int(len(list_id)*0.5)]
        list_id_val = list_id[int(len(list_id)*0.5):int(len(list_id)*0.75)]
        list_id_test = list_id[int(len(list_id)*0.75):]

    tf.config.threading.set_inter_op_parallelism_threads(input_dict['num_cpus'])
    tf.config.threading.set_intra_op_parallelism_threads(input_dict['num_cpus'])

    # get model
    maml = MAML(
        data_dict=data_dict,
        based_model=input_dict['based_model'],
        inner_lr=input_dict['inner_learning_rate'],
        outer_lr=input_dict['outer_learning_rate'],
        mode=input_dict['mode']
    )

    # potential pretrained models are copied to ./pretrained/{...}/backup
    pretrained_dir = os.path.join(PRETRAINED_DIR, dataset)
    for round in range(input_dict['num_rounds']):
        batch_task = random.sample(list_id_train, input_dict['batch_task_size'])
        print(f'\nEpoch {round+1}/{input_dict["num_rounds"]}: Meta-train on {batch_task}\n')
        maml.train(round, input_dict['num_epochs'], batch_task, list_id_val)

    # after training, testing is produced automatically
    acc, precision, recall, f1 = maml.valid(list_id_test, num_epochs=input_dict['num_epochs'], mode=TEST)

    # save log and model
    model_folder = f"{input_dict['based_model']}_{input_dict['look_back_window']}_{input_dict['inner_learning_rate']}_{input_dict['outer_learning_rate']}"
    model_dir = os.path.join(pretrained_dir, model_folder)
    # create a folder in `pretrained_dir` named model_name
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    maml.save_model(dir_=model_dir, model_name=label)

    return acc, precision, recall, f1, model_dir

if __name__ == '__main__':
    # config data_dir right here
    dataset = MULTI_FX
    data_dir = os.path.join(RAW_DATA_DIR, dataset)

    # extract columns
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            tmp_df = pd.read_csv(os.path.join(data_dir, filename)).to_numpy()[:,1:]
            columns = list(range(tmp_df.shape[1]))
            break

    # run stuff
    input_dict = parse_args()
    labels = random.sample(population=columns, k=4)
    summary = {}
    for idx, label in enumerate(labels):
        print(f'\nPredict on label {idx+1}/{len(labels)}\n')
        acc, precision, recall, f1, model_dir = run(dataset, input_dict, label)

        # log for summary a model
        summary[label] = {}
        summary[label]['acc'] = acc
        summary[label]['precision'] = precision
        summary[label]['recall'] = recall
        summary[label]['f1'] = f1

    # write summary of models
    print(f'\nWrite summary log to {model_dir}\n')
    with open(os.path.join(model_dir, 'summary.json'), 'w') as fo:
        json.dump(summary, fo)

# # test
# python -m execute.main -window 10 -model lstm_cnn -mode classification -inner_lr 0.001 -outer_lr 0.0055 -bt_size 3 -rounds 3 -epochs 3 -ncpus 26
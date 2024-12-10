import tensorflow as tf
import random
random.seed(84)
import argparse
import json

from model.meta_model import MAML
from common.constants import *
from data.dataloader import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, required=True)
    parser.add_argument('-window', '--look_back_window', type=int, required=True)
    parser.add_argument('-model', '--based_model', type=str, choices=[LSTM, CNN, LSTM_CNN, ATT], default=LSTM, required=False)
    parser.add_argument('-mode', '--mode', type=str, choices=[REG, CLF], default=CLF, required=True)
    parser.add_argument('-out_shape', '--output_shape', type=int, default=2, required=True)
    parser.add_argument('-inner_lr', '--inner_learning_rate', type=float, required=True) # we need this param in test phase
    parser.add_argument('-outer_lr', '--outer_learning_rate', type=float, default=0.001, required=False)
    parser.add_argument('-bt_size', '--batch_task_size', type=int, default=5, required=False)
    parser.add_argument('-rounds', '--num_rounds', type=int, default=100, required=False)
    parser.add_argument('-epochs', '--num_epochs', type=int, default=3, required=True) # we need this param in test phase
    parser.add_argument('-ncpus', '--num_cpus', type=int, default=8)
    # parser.add_argument('-test', '--test', type=str, required=False, help='Pass model_name here (e.g: lstm_0.001_0.005)')
    # parser.add_argument('--test', action='store_true', required=False)

    return vars(parser.parse_args())

def run(
        data_dict:dict[str, dict],
        input_dict:dict,
        model_dir:str,
        list_id_train:list[int],
        list_id_val:list[int],
        list_id_test:list[int]
    ) -> dict[str:float]:
    """train model with given hyper-param in `input_dict`

    Args:
        data_dict (dict[str, dict]): dictionary containing all tasks
        input_dict (dict): hyper-param
        model_dir (str): directory to save model and its log

    Returns:
        metric_dict_: a dictionary containing all metrics for CLF or REG problem
    """
    tf.config.threading.set_inter_op_parallelism_threads(input_dict['num_cpus'])
    tf.config.threading.set_intra_op_parallelism_threads(input_dict['num_cpus'])

    # get model
    maml = MAML(
        data_dict=data_dict,
        output_shape=input_dict['output_shape'],
        based_model=input_dict['based_model'],
        inner_lr=input_dict['inner_learning_rate'],
        outer_lr=input_dict['outer_learning_rate'],
        mode=input_dict['mode']
    )

    # meta-train
    for round in range(input_dict['num_rounds']):
        batch_task = random.sample(list_id_train, input_dict['batch_task_size'])
        print(f'\nEpoch {round+1}/{input_dict["num_rounds"]}: Meta-train on {batch_task}\n')
        maml.train(round, input_dict['num_epochs'], batch_task, list_id_val)

    # after meta-training, meta-testing is produced automatically
    if input_dict['mode'] == CLF:
        acc, precision, recall, f1 = maml.valid(list_id_test, num_epochs=input_dict['num_epochs'], mode=TEST)
        metric_dict = {
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    elif input_dict['mode'] == REG:
        mse = maml.valid(list_id_test, num_epochs=input_dict['num_epochs'], mode=TEST)
        metric_dict = {'mse': mse}

    # save log and model
    maml.save_model(dir_=model_dir, model_name=label)

    return metric_dict

def add_summary_log(summary:dict, label:int, metric_dict:dict[str:float]):
    # log for summary a model
    summary[label] = {}
    for key in metric_dict.keys():
        summary[label][key] = metric_dict[key]

def write_summary_log(summary:dict, model_dir:str):
    print(f'\nWrite summary log to {model_dir}\n')
    with open(os.path.join(model_dir, 'summary.json'), 'w') as fo:
        json.dump(summary, fo)

if __name__ == '__main__':
    # get input_dict
    input_dict = parse_args()

    # config data
    dataset = input_dict['dataset']
    dataloader = DataLoader(dataset=dataset, look_back=input_dict['look_back_window'], mode=input_dict['mode'], is_multi='multi' in dataset)

    # specify model_dir to save model and its log
    pretrained_dir = os.path.join(PRETRAINED_DIR, dataset)
    model_folder = f"{input_dict['based_model']}_{input_dict['look_back_window']}_{input_dict['inner_learning_rate']}_{input_dict['outer_learning_rate']}"
    model_dir = os.path.join(pretrained_dir, model_folder)

    # create pretrained dir for dataset
    if not os.path.exists(pretrained_dir):
        os.mkdir(pretrained_dir)
    # create a folder in `pretrained_dir` named model_dir to save model for attributes
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    summary = {}
    for idx, label in enumerate(dataloader.columns):
        print(f'\nPredict on label {idx+1}/{len(dataloader.columns)}\n')

        # get data and the given label
        data_dict = dataloader.get_multi_data(label)

        # split train, val, test
        list_id_train, list_id_val, list_id_test = dataloader.get_list_id(list(data_dict.keys()))

        # meta-train, then meta-test and obtain metrics
        metric_dict = run(data_dict, input_dict, model_dir, list_id_train, list_id_val, list_id_test)

        # log for summary a model
        add_summary_log(summary, label, metric_dict)

    # write summary of models
    write_summary_log(summary, model_dir)

# python -m execute.main -dataset USD_JPY -window 10 -model lstm_cnn -mode classification -out_shape 2 -inner_lr 0.001 -outer_lr 0.0055 -bt_size 3 -rounds 3 -epochs 3 -ncpus 8

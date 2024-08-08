import tensorflow as tf
import random
random.seed(84)
import argparse

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
    parser.add_argument('-test', '--test', type=str, required=False, help='Pass model_name here (e.g: lstm_0.001_0.005)')
    # parser.add_argument('--test', action='store_true', required=False)

    return vars(parser.parse_args())

def main():
    input_dict = parse_args()

    # data preparation
    data_dict = get_data(look_back=input_dict['look_back_window'])

    # choose randomly 50% clients for training, 25% for validating and 25% for testing (only apply for multi_fx, not multi_USDJPY)
    # if this is from `pretrained_macro_random`
    list_id = set(data_dict.keys())
    list_id_train = random.sample(sorted(list_id), 30)
    list_id_val = random.sample(sorted(list_id.difference(set(list_id_train))), 15)
    list_id_test = list(list_id.difference(set(list_id_train).union(list_id_val)))

    # # if pretrained models are from `pretrained_macro`
    # # if the data is multi_USDJPY
    # list_id = list(data_dict.keys())
    # list_id_train = list_id[:int(len(list_id)*0.5)]
    # list_id_val = list_id[int(len(list_id)*0.5):int(len(list_id)*0.75)]
    # list_id_test = list_id[int(len(list_id)*0.75):]

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
    pretrained_dir = os.path.join(PRETRAINED_DIR, 'pretrained_macro_random')
    if input_dict['test']:
        maml.meta_model = tf.keras.models.load_model(os.path.join(pretrained_dir, 'backup', f'{input_dict["test"]}.keras'))
        maml.valid(list_id_test, num_epochs=input_dict['num_epochs'])
    else:
        for round in range(input_dict['num_rounds']):
            batch_task = random.sample(list_id_train, input_dict['batch_task_size'])
            print(f'\nEpoch {round+1}/{input_dict["num_rounds"]}: Meta-train on {batch_task}\n')
            maml.train(round, input_dict['num_epochs'], batch_task, list_id_val)

        model_name = f"{input_dict['based_model']}_{input_dict['look_back_window']}_{input_dict['inner_learning_rate']}_{input_dict['outer_learning_rate']}"
        maml.save_model(dir_=pretrained_dir, model_name=model_name)

if __name__ == '__main__':
    main()

# # test
# python -m execute.main -test lstm_30_0.005_0.0055 -window 20 -mode classification -inner_lr 0.001 -epochs 3 -ncpus 16
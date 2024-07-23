from keras import optimizers
import random

from model.meta_model import MAML
from common.constants import *
from data.pre_process import get_data
from .parser import parse_args

def main():
    input_dict = parse_args()

    # data preparation
    data_dict = get_data(look_back=input_dict['look_back_window'])
    list_id = list(data_dict.keys())
    list_id_train = list_id[:int(len(list_id)*0.7)]
    list_id_val = list_id[int(len(list_id)*0.7):int(len(list_id)*0.85)]
    list_id_test = list_id[int(len(list_id)*0.85):]

    # get model
    maml = MAML(
        data_dict=data_dict,
        based_model=input_dict['based_model'],
        inner_opt=optimizers.Adam(learning_rate=input_dict['inner_learning_rate']),
        outer_opt=optimizers.Adam(learning_rate=input_dict['outer_learning_rate']),
        mode=input_dict['mode']
    )

    for round in range(input_dict['num_rounds']):
        batch_task = random.sample(list_id_train, input_dict['batch_task_size'])
        print(f'\nEpoch {round+1}/{input_dict["num_rounds"]}: Meta-train on {batch_task}\n')
        loss, acc = maml.train(round, input_dict['num_epochs'], batch_task, list_id_val)

if __name__ == '__main__':
    main()

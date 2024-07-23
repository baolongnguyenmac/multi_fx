import argparse

from common.constants import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-window', '--look_back_window', type=int, default=10)
    parser.add_argument('-model', '--based_model', type=str, choices=[LSTM, CNN, LSTM_CNN, ATT], default=LSTM)
    parser.add_argument('-mode', '--mode', type=str, choices=[REG, CLF], default=CLF)
    parser.add_argument('-inner_lr', '--inner_learning_rate', type=float, default=0.001)
    parser.add_argument('-outer_lr', '--outer_learning_rate', type=float, default=0.001)
    parser.add_argument('-bt_size', '--batch_task_size', type=int, default=3)
    parser.add_argument('-rounds', '--num_rounds', type=int, default=100)
    parser.add_argument('-epochs', '--num_epochs', type=int, default=2)

    return vars(parser.parse_args())

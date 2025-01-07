# viết 1 class
# nhận vào loại model (lstm, lstm+cnn, attention)
# nhận vào kiểu data (is_multi), phải viết lại dataloader
# chạy đơn lẻ các model trên dữ liệu
# không finetune
# lưu log

import pandas as pd
import numpy as np
import keras
import argparse
import json

from model.based_model import get_model
from data.dataloader import DataLoader
from common.constants import *
from common.common import compute_metrics

class DataLoaderAblation(DataLoader):
    def __init__(self, dataset, look_back = None, batch_size = -1, test_size = 0.2, mode = CLF, is_multi = False):
        super().__init__(dataset, look_back, batch_size, test_size, mode, is_multi)

    def get_ablation_data(self, label:int=-1):
        data_dir = self.data_dir if self.is_multi else os.path.join(self.data_dir, 'all')
        data_dict = {}

        list_file = os.listdir(data_dir)
        for cid, file_name in enumerate(list_file):
            if file_name.endswith('.csv'):
                # read dataset, drop the datetime/index column
                df = pd.read_csv(os.path.join(data_dir, file_name)).to_numpy()[:,1:]

                # normalize data
                transformed_df, mean_, std_ = self._normalize_data(df=df, label=label, test_size=0.2)

                # create dataset with lookback window
                X, y = self._create_dataset(df=transformed_df, label=label)

                # add to data_dict
                tmp_dict = self.split_support_query(X, y, mean_, std_)
                num_train = tmp_dict['support_X'].shape[0]
                # only use 60% data for training
                tmp_dict['support_X'] = tmp_dict['support_X'][:int(num_train*0.75)]
                tmp_dict['support_y'] = tmp_dict['support_y'][:int(num_train*0.75)]
                data_dict[cid] = tmp_dict

        return data_dict

class Ablation:
    def compile_and_fit(self, data_task:dict, model_name:str, n_epochs:int, lr:float, batch_size:int=32):
        X_train, X_test, y_train, y_test = data_task['support_X'], data_task['query_X'], data_task['support_y'], data_task['query_y']
        model = get_model(model_name, X_train.shape[1:], 2, CLF)
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[callback])
        return model, history

    def _run_model(self, param_dict:dict):
        # train
        model, history = self.compile_and_fit(param_dict['data_task'], param_dict['model_name'], param_dict['n_epochs'], param_dict['lr'], param_dict['batch_size'])

        # compute metric
        y_pred = model(param_dict['data_task']['query_X'])
        y_pred = np.argmax(y_pred.numpy(), axis=1)

        acc, precision, recall, f1 = compute_metrics(y_pred, param_dict['data_task']['query_y'])
        metric_dict = {
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        return metric_dict, history

    def add_log(self, history, metric_dict:dict):
        log = {}
        log['accuracy'] = history.history['accuracy']
        log['val_accuracy'] = history.history['val_accuracy']

        for key in metric_dict.keys():
            log[key] = metric_dict[key]

        return log

    def save_log(self, log:dict, filename:str):
        # save log
        filename = os.path.join(filename)
        with open(filename, 'w') as fo:
            json.dump(log, fo)

    def run_model(self, dataset:str, model_name:str, model_dir:str, look_back:int=30, n_epochs:int=100, lr:float=0.001, batch_size:int=32):
        dataloader = DataLoaderAblation(dataset=dataset, look_back=look_back, mode=CLF, is_multi='multi' in dataset)
        log = {}
        for label in dataloader.columns:
            log[label] = {}
            data_dict = dataloader.get_ablation_data(label=label)
            if 'multi' not in dataset:
                # metrics = (acc_, precision_, recall_, f1_)
                metric_dict, history = self._run_model({
                    'data_task': data_dict[0],
                    'model_name': model_name,
                    'n_epochs': n_epochs,
                    'lr': lr,
                    'batch_size': batch_size
                })
                # add log for a label
                for key in metric_dict.keys():
                    log[label][key] = metric_dict[key]

                # save log for a label
                self.save_log(
                    log = self.add_log(history, metric_dict),
                    filename = os.path.join(model_dir, f"{label}.json"))
            else:
                # init metrics
                list_metrics = {}
                for m in ["acc", "precision", "recall", "f1"]:
                    list_metrics[m] = []

                tmp_log = {}

                for task in data_dict.keys():
                    # metrics = (acc_, precision_, recall_, f1_)
                    metric_dict, history = self._run_model({
                        'data_task': data_dict[task],
                        'model_name': model_name,
                        'n_epochs': n_epochs,
                        'lr': lr,
                        'batch_size': batch_size
                    })
                    # add metric of a task
                    for key in metric_dict.keys():
                        list_metrics[key].append(metric_dict[key])

                    # add log for a task for a label
                    tmp_log[task] = self.add_log(history, metric_dict)

                # save log for a label (contain many tasks)
                self.save_log(tmp_log, os.path.join(model_dir, f"{label}.json"))

                # aggregate metrics
                for m in ["acc", "precision", "recall", "f1"]:
                    log[label][m] = f"{np.mean(list_metrics[m]):.5f}±{np.std(list_metrics[m]):.5f}"

        # save log for all label
        filename = os.path.join(model_dir, "summary.json")
        self.save_log(log, filename)

if __name__ == '__main__':
    # get dataset from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', type=str, required=True)
    input_dict = vars(parser.parse_args())

    # config dataset and model
    dataset = input_dict['dataset']

    # tạo folder để lưu log
    # create ablation_dir if needed
    ablation_dir = os.path.join(PRETRAINED_DIR, 'ablation')
    if not os.path.exists(ablation_dir):
        os.mkdir(ablation_dir)

    # create a folder for each dataset in ablation_dir
    dataset_dir = os.path.join(ablation_dir, dataset)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    for model_name in [LSTM, LSTM_CNN, ATT]:
        # init an Ablation object
        ab = Ablation()

        # config hyper-param
        look_back = 30
        n_epochs = 101
        lr = 0.001
        batch_size = 32

        # create a folder of each model
        model_dir = os.path.join(dataset_dir, f"{model_name}_{look_back}_{n_epochs}_{lr}_{batch_size}")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        # run model
        ab.run_model(dataset, model_name, model_dir, look_back, n_epochs, lr, batch_size)

# bỏ callback và phân tích bằng tay

# python -m execute.ablation -dataset multi_fx
# python -m execute.ablation -dataset USD_JPY
# python -m execute.ablation -dataset ett
# python -m execute.ablation -dataset wth
import tensorflow as tf
from keras import optimizers, models, metrics
# from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from copy import deepcopy
import json
import os

from .based_model import get_model
from common.constants import *
from common.common import compute_metrics

class MAML:
    def __init__(
        self,
        data_dict:dict[str, dict[str, np.ndarray]],
        based_model:str,
        inner_lr:float,
        outer_lr:float,
        mode:str=CLF
    ):
        # get data
        self.data_dict:dict[str, dict[str, np.ndarray]] = data_dict

        # init model stuff
        self.meta_model:models.Model = get_model(based_model, data_dict[0]['support_X'][0].shape[1:], mode)
        self.outer_opt:optimizers.Optimizer = optimizers.Adam(learning_rate=outer_lr)
        self.inner_lr:float = inner_lr

        # init log
        self.info = {}
        self.info['look_back'] = data_dict[0]['support_X'][0].shape[1]
        self.info['based_model'] = based_model
        self.info['inner_lr'] = inner_lr
        self.info['outer_lr'] = outer_lr
        self.info['train_loss'] = []
        self.info['train_acc'] = []
        self.info['val_loss'] = []
        self.info['val_acc'] = []
        self.info['val_std_loss'] = []
        self.info['val_std_acc'] = []

        # init metrics
        self.mode = mode
        if mode == REG:
            self.loss_fn = metrics.mean_squared_error
        elif mode == CLF:
            self.loss_fn = metrics.binary_crossentropy
            # self.acc_fn = metrics.Accuracy()

    def inner_training_step(self, model:models.Model, X:list[np.ndarray], y:list[np.ndarray], num_epochs:int=2):
        """fast adapt on support set (inner step)

        Args:
            weights: weights of meta model
            X, y: data, divided into batches

        Returns:
            weights of new model
        """

        inner_opt = optimizers.Adam(learning_rate=self.inner_lr)
        for _ in range(num_epochs):
            for batch_X, batch_y in zip(X, y):
                with tf.GradientTape() as tape:
                    pred = model(batch_X)
                    loss = self.loss_fn(batch_y, pred)

                # Calculate the gradients for the variables
                gradients = tape.gradient(loss, model.trainable_variables)
                # Apply the gradients and update the optimizer
                inner_opt.apply_gradients(zip(gradients, model.trainable_variables))

        return model.get_weights()

    def outer_compute(self, batch_id_train:list[int], task_weights:dict[list[np.ndarray]]):
        """outer compute on query set (outer step)

        Args:
            batch_id_train: sample a batch of task
            task_weights: list of weights, which have adapted to support sets

        Returns:
            list of accuracy and loss
        """
        sum_task_losses = 0.

        tasks_metrics = {
            'acc':[],
            'loss':[],
            'precision':[],
            'recall':[],
            'f1':[]
        }

        for task_id in batch_id_train:
            pred_values = np.empty((0,1))
            true_values = np.empty((0,1))
            task_loss = 0.

            # Get each saved optimized weight.
            self.meta_model.set_weights(task_weights[task_id])

            X = self.data_dict[task_id]['query_X']
            y = self.data_dict[task_id]['query_y']
            for batch_X, batch_y in zip(X, y):
                pred = self.meta_model(batch_X)
                loss = self.loss_fn(batch_y, pred)
                task_loss += tf.reduce_sum(loss)

                # store prediction and true label for computing acc
                if self.mode == CLF:
                    pred = (pred.numpy()>=0.5)*1.
                    pred_values = np.vstack((pred_values, pred))
                    true_values = np.vstack((true_values, batch_y))

            # compute sum loss for back-propagation
            sum_task_losses += task_loss

            # compute acc + loss of a task
            if self.mode == CLF:
                acc_, precision_, recall_, f1_ = compute_metrics(pred_values, true_values)
                tasks_metrics['acc'].append(acc_)
                tasks_metrics['precision'].append(precision_)
                tasks_metrics['recall'].append(recall_)
                tasks_metrics['f1'].append(f1_)
                task_loss /= len(true_values)
                print(f'Outer compute on task {task_id}: loss={task_loss:.5f}\t acc={acc_:.5f}')
            elif self.mode == REG:
                task_loss /= len(X)
                print(f'Outer compute on task {task_id}: loss={task_loss:.5f}')

            tasks_metrics['loss'].append(task_loss)

        tasks_metrics['loss'], tasks_metrics['std_loss'] = compute_metrics(tasks_metrics['loss'])
        print(f"\n\tMean loss:\t{tasks_metrics['loss']:.5f} ± {tasks_metrics['std_loss']:.5f}")
        if self.mode == CLF:
            tasks_metrics['acc'], tasks_metrics['std_acc'] = compute_metrics(tasks_metrics['acc'])
            tasks_metrics['precision'], tasks_metrics['std_precision'] = compute_metrics(tasks_metrics['precision'])
            tasks_metrics['recall'], tasks_metrics['std_recall'] = compute_metrics(tasks_metrics['recall'])
            tasks_metrics['f1'], tasks_metrics['std_f1'] = compute_metrics(tasks_metrics['f1'])

            print(f"\tMean acc:\t{tasks_metrics['acc']:.5f} ± {tasks_metrics['std_acc']:.5f}")
            print(f"\tMean precision:\t{tasks_metrics['precision']:.5f} ± {tasks_metrics['std_precision']:.5f}")
            print(f"\tMean recall:\t{tasks_metrics['recall']:.5f} ± {tasks_metrics['std_recall']:.5f}")
            print(f"\tMean f1:\t{tasks_metrics['f1']:.5f} ± {tasks_metrics['std_f1']:.5f}")
        print('================================')

        return sum_task_losses, tasks_metrics

    # def compute_metrics(self, pred_values:list[float], true_values=None):
    #     if true_values is None:
    #         # compute metrics across tasks or simply take the average of a list of metric
    #         return float(np.mean(pred_values)), float(np.std(pred_values))
    #     else:
    #         # compute metrics in a task
    #         self.acc_fn.update_state(true_values, pred_values)
    #         precision, recall, f1, _ = precision_recall_fscore_support(true_values, pred_values, average='macro')
    #         return float(self.acc_fn.result()), float(precision), float(recall), float(f1)

    def train(self, round:int, num_epochs:int, batch_id_train:list[int], list_id_val:list[int]=None):
        """train a batch of task

        Args:
            round: current round
            num_epochs: number of epochs for inner opt
            batch_id_train: sample a batch of task
            list_id_val: list task for validation

        Returns:
            accuracy (if mode=classification) and loss
        """
        # print(f'\n========= Epoch {epoch}: Meta-train on {list_id_train} =========\n')
        task_weights = {}

        # Get current model's weight to make sure that model's weight is reset in beginning
        # of each inner loop.
        meta_weights = self.meta_model.get_weights()

        # Inner loops.
        # Loop through all support dataset and update model weight.
        for task_id in batch_id_train:
            print(f'Inner optimize on task {task_id}')

            X = self.data_dict[task_id]['support_X']
            y = self.data_dict[task_id]['support_y']

            # Get starting initialized weight.
            self.meta_model.set_weights(meta_weights)

            # Save optimized weight of each support task.
            task_weights[task_id] = self.inner_training_step(self.meta_model, X, y, num_epochs)

        print()

        # Calculate loss of each optimized weight on query training dataset set.
        with tf.GradientTape() as tape:
            sum_loss, tasks_metrics = self.outer_compute(batch_id_train, task_weights)
            self.info['train_loss'].append(tasks_metrics['loss'])
            self.info['train_acc'].append(tasks_metrics['acc'])

        # Get starting initialized weight. 
        self.meta_model.set_weights(meta_weights)

        # Back-propagation of outer loop.
        grads = tape.gradient(sum_loss, self.meta_model.trainable_variables)
        self.outer_opt.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        if (round+1)%5 == 0 or round==0:
            self.valid(list_id_val, num_epochs)

    def valid(self, list_id_val:list[int], num_epochs:int=2, mode:str=VAL):
        print('\nValidation\n')
        model:models.Model = deepcopy(self.meta_model)
        meta_weights = model.get_weights()

        task_weights = {}

        for task_id in list_id_val:
            print(f'Adapt on task {task_id}')

            X = self.data_dict[task_id]['support_X']
            y = self.data_dict[task_id]['support_y']

            # Get starting initialized weight.
            model.set_weights(meta_weights)
            task_weights[task_id] = self.inner_training_step(model, X, y, num_epochs)

        print()
        _, tasks_metrics = self.outer_compute(list_id_val, task_weights)

        if mode == VAL:
            self.info['val_loss'].append(tasks_metrics['loss'])
            self.info['val_acc'].append(tasks_metrics['acc'])
            self.info['val_std_loss'].append(tasks_metrics['std_loss'])
            self.info['val_std_acc'].append(tasks_metrics['std_acc'])

        self.info[f'{mode}_accuracy'] = f"{tasks_metrics['acc']:.5f} ± {tasks_metrics['std_acc']:.5f}"
        self.info[f'{mode}_precision'] = f"{tasks_metrics['precision']:.5f} ± {tasks_metrics['std_precision']:.5f}"
        self.info[f'{mode}_recall'] = f"{tasks_metrics['recall']:.5f} ± {tasks_metrics['std_recall']:.5f}"
        self.info[f'{mode}_f1'] = f"{tasks_metrics['f1']:.5f} ± {tasks_metrics['std_f1']:.5f}"

    def save_model(self, dir_:str, model_name:str):
        print('\nSave model\n')
        json_file_path = os.path.join(dir_, f'{model_name}.json')
        model_file_path = os.path.join(dir_, f'{model_name}.keras')
        with open(json_file_path, 'w') as fo:
            json.dump(self.info, fo)
        self.meta_model.save(model_file_path)

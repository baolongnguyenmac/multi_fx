import tensorflow as tf
from keras import optimizers, models, metrics
import numpy as np
from copy import deepcopy
import json
import os

from .based_model import get_model
from common.constants import *

class MAML:
    def __init__(
        self,
        data_dict:dict[str, dict[str, np.ndarray]],
        based_model:str,
        inner_lr:float,
        outer_lr:float,
        mode:str=CLF
    ):
        self.meta_model:models.Model = get_model(based_model, data_dict[0]['support_X'][0].shape[1:], mode)
        self.data_dict:dict[str, dict[str, np.ndarray]] = data_dict
        # self.inner_opt:optimizers.Optimizer = inner_opt
        self.outer_opt:optimizers.Optimizer = optimizers.Adam(learning_rate=outer_lr)
        self.inner_lr:float = inner_lr

        self.info = {}
        self.info['look_back'] = data_dict[0]['support_X'][0].shape[1]
        self.info['based_model'] = based_model
        self.info['inner_lr'] = inner_lr
        self.info['outer_lr'] = outer_lr
        self.info['train_loss'] = []
        self.info['train_acc'] = []
        self.info['val_loss'] = []
        self.info['val_acc'] = []

        if mode == REG:
            self.loss_fn = metrics.mean_squared_error
        elif mode == CLF:
            self.loss_fn = metrics.binary_crossentropy
            self.acc_fn = metrics.Accuracy()

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
        batch_acc = []
        batch_loss = []

        for task_id in batch_id_train:
            #Get each saved optimized weight.
            self.meta_model.set_weights(task_weights[task_id])

            X = self.data_dict[task_id]['query_X']
            y = self.data_dict[task_id]['query_y']
            for batch_X, batch_y in zip(X, y):
                pred = self.meta_model(batch_X)
                loss = self.loss_fn(batch_y, pred)

                try:
                    tmp_pred = pred.numpy()
                    tmp_pred[tmp_pred >= 0.5] = 1
                    tmp_pred[tmp_pred < 0.5] = 0
                    self.acc_fn.update_state(batch_y, tmp_pred)
                    batch_acc.append(tf.get_static_value(self.acc_fn.result()))
                except:
                    pass

                batch_loss.append(loss)

            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                if len(batch_acc) != 0:
                    print(f'Outer compute on task {task_id}: loss={np.mean(batch_loss[-1]).item():.5f}\t acc={np.mean(batch_acc[-1]).item():.5f}')
                else:
                    print(f'Outer compute on task {task_id}: loss={np.mean(batch_loss[-1]).item():.5f}')

        # Calculate sum loss
        # Calculate mean loss only for visualizing.
        sum_loss = tf.reduce_sum([tf.reduce_sum(l) for l in batch_loss])
        mean_loss = tf.get_static_value(tf.reduce_mean([tf.reduce_mean(l) for l in batch_loss]))
        mean_acc = tf.get_static_value(tf.reduce_mean([tf.reduce_mean(a) for a in batch_acc]))

        print(f'\n\tMean loss:\t{mean_loss:.5f}')
        if len(batch_acc) != 0:
            print(f'\tMean acc:\t{mean_acc:.5f}')
        print('================================')

        return (sum_loss, mean_loss.item(), mean_acc.item()) if len(batch_acc)!= 0 else (sum_loss, mean_loss.item(), -1)

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
            sum_loss, mean_loss, mean_acc = self.outer_compute(batch_id_train, task_weights)
            self.info['train_loss'].append(mean_loss)
            self.info['train_acc'].append(mean_acc)

        # Get starting initialized weight. 
        self.meta_model.set_weights(meta_weights)

        # Back-propagation of outer loop.
        grads = tape.gradient(sum_loss, self.meta_model.trainable_variables)
        self.outer_opt.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        if (round+1)%5 == 0 or round==0:
            self.valid(list_id_val, num_epochs)

        return mean_loss, mean_acc

    def valid(self, list_id_val:list[int], num_epochs:int=2):
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
        _, mean_loss, mean_acc = self.outer_compute(list_id_val, task_weights)
        self.info['val_loss'].append(mean_loss)
        self.info['val_acc'].append(mean_acc)

        return mean_loss, mean_acc

    def save_model(self, model_name:str='model'):
        print('\nSave model\n')
        with open(f'./pretrained/{model_name}.json', 'w') as fo:
            json.dump(self.info, fo)
        self.meta_model.save(os.path.join(PRETRAINED_DIR, f'{model_name}.keras'))

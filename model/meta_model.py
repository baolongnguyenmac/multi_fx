import tensorflow as tf
from keras import optimizers, models, metrics
import numpy as np

from .based_model import get_model
from common.constants import *
from data.pre_process import get_data

class MAML:
    def __init__(
        self,
        data_dict:dict[str, dict[str, np.ndarray]],
        based_model:str,
        inner_opt:optimizers.Optimizer,
        outer_opt:optimizers.Optimizer,
        mode:str=CLF
    ):
        self.meta_model:models.Model = get_model(based_model, data_dict[0]['support_X'][0].shape[1:], mode)
        self.data_dict:dict[str, dict[str, np.ndarray]] = data_dict
        self.inner_opt:optimizers.Optimizer = inner_opt
        self.outer_opt:optimizers.Optimizer = outer_opt
        if mode == REG:
            self.loss_fn = metrics.mean_squared_error
        elif mode == CLF:
            self.loss_fn = metrics.binary_crossentropy
            self.acc_fn = metrics.Accuracy()

    #Training step of each batch.
    def train_on_batch(self, epoch:int, list_id_train:list[int], list_id_val:list[int]=None):
        print(f'\n========= Epoch {epoch}: Meta-train on {list_id_train} =========\n')
        batch_acc = []
        batch_loss = []
        task_weights = []

        # Get current model's weight to make sure that model's weight is reset in beginning
        # of each inner loop.    
        meta_weights = self.meta_model.get_weights()

        # Inner loops.
        # Loop through all support dataset and update model weight.
        for key in list_id_train:
            print(f'Inner optimize on task {key}')
            # Get starting initialized weight.
            self.meta_model.set_weights(meta_weights)

            X = self.data_dict[key]['support_X']
            y = self.data_dict[key]['support_y']
            for batch_X, batch_y in zip(X, y):
                with tf.GradientTape() as tape:
                    pred = self.meta_model(batch_X)
                    loss = self.loss_fn(batch_y, pred)

                # Calculate the gradients for the variables
                gradients = tape.gradient(loss, self.meta_model.trainable_variables)
                # Apply the gradients and update the optimizer
                self.inner_opt.apply_gradients(zip(gradients, self.meta_model.trainable_variables))

            # Save optimized weight of each support task. 
            task_weights.append(self.meta_model.get_weights())

        print()
        # Calculate loss of each optimized weight on query training dataset set.
        with tf.GradientTape() as tape:
            for key in list_id_train:
                print(f'Outer optimize on task {key}')
                #Get each saved optimized weight.
                self.meta_model.set_weights(task_weights[key])

                X = self.data_dict[key]['query_X']
                y = self.data_dict[key]['query_y']
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

            # Calculate sum loss
            # Calculate mean loss only for visualizing.
            sum_loss = tf.reduce_sum([tf.reduce_sum(l) for l in batch_loss])
            mean_loss = tf.get_static_value(tf.reduce_mean([tf.reduce_mean(l) for l in batch_loss]))
            mean_acc = tf.get_static_value(tf.reduce_mean([tf.reduce_mean(a) for a in batch_acc]))

            print(f'\tOuter loss of epoch {epoch}:\t{mean_loss:.5f}')
            if len(batch_acc) != 0:
                print(f'\tAccuracy on query:\t{mean_acc:.5f}')

        # Get starting initialized weight. 
        self.meta_model.set_weights(meta_weights)

        # Back-propagation of outer loop.
        grads = tape.gradient(sum_loss, self.meta_model.trainable_variables)
        self.outer_opt.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        return (mean_loss, mean_acc) if len(batch_acc) != 0 else (mean_loss, 0)

        # # Calculate loss of each optimized weight on query training dataset set.
        # with tf.GradientTape() as tape:
        #     for key in list_id_train:
        #         #Get each saved optimized weight.
        #         self.meta_model.set_weights(task_weights[key])

        #         X = self.data_dict[key]['query_X']
        #         y = self.data_dict[key]['query_y']

        #         for batch_X, batch_y in zip(X, y):
        #             pred = self.meta_model(batch_X)
        #             loss = self.loss_fn(batch_y, pred)

        #         try:
        #             self.acc_fn.update_state(y, pred)
        #             batch_acc.append(tf.get_static_value(self.acc_fn.result()))
        #         except:
        #             pass

        #         batch_loss.append(loss)

        #     # Calculate sum loss
        #     # Calculate mean loss only for visualizing.
        #     sum_loss = tf.reduce_sum(batch_loss)
        #     mean_loss = tf.get_static_value(tf.reduce_mean(batch_loss))
        #     mean_acc = tf.get_static_value(tf.reduce_mean(batch_acc))

        # # Get starting initialized weight. 
        # self.meta_model.set_weights(meta_weights)

        # # Back-propagation of outer loop.
        # grads = tape.gradient(sum_loss, self.meta_model.trainable_variables)
        # self.outer_opt.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        # return (mean_loss, mean_acc) if len(batch_acc) != 0 else (mean_loss, 0)



from keras import optimizers

from model.meta_model import MAML
from common.constants import *
from data.pre_process import get_data

data_dict = get_data(look_back=20)
list_id = list(data_dict.keys())
num_id = len(list_id)

# config list_id_train --> batch
maml = MAML(
    data_dict=data_dict,
    based_model=LSTM,
    inner_opt=optimizers.Adam(learning_rate=0.001),
    outer_opt=optimizers.Adam(learning_rate=0.001),
    mode=CLF
)

for i in range(10):
    print(f'Epoch {i+1}/10')
    loss, acc = maml.train_on_batch(i, list_id[:2])
    print(loss, acc)
    print('================\n')

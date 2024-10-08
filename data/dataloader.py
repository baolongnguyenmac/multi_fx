import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from common.constants import *

class DataLoader:
    def __init__( self, dataset:str, look_back:int=None, batch_size:int=32, test_size:float=0.8, mode:str=CLF) -> None:
        """initialize for dataloader. in case of NHITS, we need nothing but the `dataset`

        Args:
            dataset (str): name of dataset
            look_back (int, optional): number of historical data-points used to predict, None means loading data for NHITS. Defaults to None.
            batch_size (int, optional): batch size. Defaults to 32. If `batch_size=-1`, then we don't divide data into batches
            test_size (float, optional): the amount of data in query set. Defaults to 0.8.
            mode (str, optional): either classification problem or regression problem. Defaults to CLF.
        """
        self.look_back = look_back
        self.batch_size = batch_size
        self.test_size = test_size
        self.dataset = dataset
        self.data_dir = os.path.join(RAW_DATA_DIR, dataset)
        self.mode = mode

        # extract columns by reading 1 *.csv file
        # this is not used for NHITS, the one for NHITS is written at `get_data`
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                tmp_df = pd.read_csv(os.path.join(self.data_dir, filename)).to_numpy()[:,1:]
                self.columns = list(range(tmp_df.shape[1]))
                break

    def _create_label(self, df:np.ndarray, label:int) -> np.ndarray:
        """create label for a given dataframe
        if this is REG problem, we remove that column in the given dataframe

        Args:
            df (np.ndarray): given dataframe
            label (int): index of column chosen to be the label

        Returns:
            np.ndarray: the label column
        """
        label_col = df[:,label]
        if self.mode == CLF:
            label_col = label_col[1:] - label_col[:-1]
            label_col = (label_col>0)*1.
            label_col = np.append([-1], label_col) # the first value is assigned to -1, it's removed anyway
        elif self.mode == REG:
            # if it is regression problem, we have to drop the label column in df
            df = np.delete(df, label, axis=1)
        return label_col

    def _normalize_data(self, df:np.ndarray, label:int=-1) -> np.ndarray:
        scaler = StandardScaler()
        return scaler.fit_transform(df), scaler.mean_[label], scaler.scale_[label]

    def _create_dataset(self, df:np.ndarray, label:int, include_current:bool=False)->tuple[np.ndarray, np.ndarray]:
        """create dataset with lookback window
        there are 3 scenarios:
            [x] use `x_{t-L:t}` to predict `y_{t+1}` ----> since we have to control `x_{t+1}` so `y_{t+1}` can be a good value, i don't think we need this scenario
            [v] use `x_t` to predict `y_t` ----> set `lookback=0` and `include_current=False`
            [v] use `x_{t-L:t}` to predict `y_t` ----> set `include_current=True`
        i have to manually config these scenarios in the code (instead of passing parameter)

        Args:
            df (np.ndarray): original dataset
            label (int): index of label column
            include_current (bool): True if we use `x_{t-L:t}` to predict `y_t`, default: False

        Returns:
            tuple[np.ndarray, np.ndarray]: X and y
        """
        include_current = False if self.look_back==0 else include_current
        size = df.shape[0] - self.look_back + include_current
        y = self._create_label(df, label)
        X = np.array([df[i : i+self.look_back+include_current] for i in range(size)])
        y = y[self.look_back-include_current:]
        return X, y.reshape(-1,1)

    def get_multi_data(self, label:int=-1) -> dict[str, dict]:
        """
            read data from files and return a dict whose each value contains data from a file
            this function is used for ML algorithm only

            label (int, optional): The label to be predict the movement. Defaults to -1 (the final feature).
        """
        cid = 0
        data_dict:dict[str, dict[str, pd.DataFrame]] = {}
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv'):
                # read dataset, drop the datetime/index column
                df = pd.read_csv(os.path.join(self.data_dir, file_name)).to_numpy()[:,1:]

                # normalize data
                transformed_df, mean_, std_ = self._normalize_data(df, label)

                # create dataset with lookback window
                X, y = self._create_dataset(transformed_df, self.look_back, label)

                tmp_dict = {}
                tmp_dict['support_X'], tmp_dict['query_X'], tmp_dict['support_y'], tmp_dict['query_y'] = train_test_split(X, y, test_size=self.test_size, shuffle=False)
                if self.batch_size != -1: # do not divide data into batches
                    for key in tmp_dict.keys():
                        num_batch = int(np.ceil(tmp_dict[key].shape[0]/self.batch_size))
                        tmp_dict[key] = np.array_split(tmp_dict[key], num_batch)
                tmp_dict['mean'] = mean_
                tmp_dict['std'] = std_
                data_dict[cid] = tmp_dict
                cid += 1

        return data_dict

    def get_data(self, label:int=-1) -> tuple[pd.DataFrame, float, float]:
        """
            this function is used for getting data for nhits model
            return a pd.DataFrame as format of nhits model
        """
        # path to a file containing all data
        data_path = os.path.join(self.data_dir, 'all')

        for file_name in os.listdir(data_path):
            # data_path should contain only 1 *.csv file
            if file_name.endswith('.csv'):
                # read dataset, drop the datetime/index column
                df = pd.read_csv(os.path.join(data_path, file_name))

                # save the columns to config df for NHITS
                if self.mode == CLF:
                    columns = list(df.columns[1:])
                elif self.mode == REG:
                    columns = list(df.columns[1:])
                    columns.pop(label)

                df = df.to_numpy()[:,1:]

                # normalize data
                transformed_df, mean_, std_ = self._normalize_data(df, label)

                # create label
                y = self._create_label(transformed_df, label)

                # concat X, y
                transformed_df = np.concatenate([transformed_df, y.reshape(-1,1)], axis=1)
                transformed_df = pd.DataFrame(transformed_df, columns=columns+['y'])
                transformed_df['unique_id'] = 0
                transformed_df['ds'] = pd.date_range(start='2000-01-01', periods=len(transformed_df), freq='h') # re-index all data
                return transformed_df, mean_, std_

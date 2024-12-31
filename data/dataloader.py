import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import random
random.seed(84)

from common.constants import *

class DataLoader:
    def __init__(
        self,
        dataset:str,
        look_back:int=None,
        batch_size:int=32,
        test_size:float=0.8,
        mode:str=CLF,
        is_multi:bool=False
    ) -> None:
        """initialize for dataloader. in case of NHITS, we need nothing but the `dataset`

        Args:
            dataset (str): name of dataset
            look_back (int, optional): number of historical data-points used to predict, None means loading data for NHITS. Defaults to None.
            batch_size (int, optional): batch size. Defaults to 32. If `batch_size=-1`, then we don't divide data into batches
            test_size (float, optional): the amount of data in query set. Defaults to 0.8.
            mode (str, optional): either classification problem or regression problem. Defaults to CLF.
            is_multi (bool, optional): whether the dataset contains many sub-datasets or not
        """
        self.look_back = look_back
        self.batch_size = batch_size
        self.test_size = test_size
        self.dataset = dataset
        self.data_dir = os.path.join(RAW_DATA_DIR, dataset)
        self.mode = mode
        self.is_multi = is_multi

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
            label_col = np.append([-1.], label_col) # the first value is assigned to -1, it's removed anyway
        elif self.mode == REG:
            # if it is regression problem, we don't have to to anything
            pass
        return label_col

    # normalize: fit on 80% data and transform on 20% for nhits, 60% and 40% for meta
    def _normalize_data(self, df:np.ndarray, label:int=-1, test_size:float=0.4):
        train_, test_ = train_test_split(df, test_size=test_size, shuffle=False)
        scaler = StandardScaler()
        train_ = scaler.fit_transform(train_)
        test_ = scaler.transform(test_)
        mean_, std_ = scaler.mean_[label], scaler.scale_[label]
        return np.concatenate([train_, test_], axis=0), mean_, std_

    def _create_dataset(self, df:np.ndarray, label:int)->tuple[np.ndarray, np.ndarray]:
        size = df.shape[0] - self.look_back
        y = self._create_label(df, label)
        X = np.array([df[i : i+self.look_back] for i in range(size)])
        y = y[self.look_back:]
        return X, y.reshape(-1,1)

    def get_multi_data(self, label:int=-1) -> dict[str, dict]:
        """
            read data from files and return a dict whose each value contains data from a file
            this function is used for ML algorithm only

            label (int, optional): The label to be predict the movement. Defaults to -1 (the final feature).
        """
        data_dict:dict[str, dict[str, pd.DataFrame]] = {}
        list_file = os.listdir(self.data_dir)
        for cid, file_name in enumerate(list_file):
            if file_name.endswith('.csv'):
                # read dataset, drop the datetime/index column
                df = pd.read_csv(os.path.join(self.data_dir, file_name)).to_numpy()[:,1:]

                # normalize data
                transformed_df, mean_, std_ = self._normalize_data(df, label)

                # create dataset with lookback window
                X, y = self._create_dataset(df=transformed_df, label=label)

                if self.is_multi:
                    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5, shuffle=False)

                    tmp_train_dict = self.split_support_query(train_X, train_y, mean_, std_)
                    tmp_test_dict = self.split_support_query(test_X, test_y, mean_, std_)

                    data_dict[cid] = tmp_train_dict
                    data_dict[cid+len(list_file)] = tmp_test_dict
                else:
                    tmp_dict = self.split_support_query(X, y, mean_, std_)
                    data_dict[cid] = tmp_dict

        return data_dict

    def split_support_query(self, df_X:pd.DataFrame, df_y:pd.DataFrame, mean_:float=None, std_:float=None):
        """split df into support and query set, by batch

        Args:
            df_X (pd.DataFrame): original df
            df_y (pd.DataFrame): original df
            mean_ (float, optional): mean of label (only used in case of regression). Defaults to None.
            std_ (float, optional): std of label (only used in case of regression). Defaults to None.

        Returns:
            dict[str, something]: contains support and query data, mean and std sometime
        """
        data_dict:dict[str:pd.DataFrame] = {}
        data_dict['support_X'], data_dict['query_X'], data_dict['support_y'], data_dict['query_y'] = train_test_split(df_X, df_y, test_size=self.test_size, shuffle=False)
        if self.batch_size != -1: # divide data into batches
            for key in data_dict.keys():
                num_batch = int(np.ceil(data_dict[key].shape[0]/self.batch_size))
                data_dict[key] = np.array_split(data_dict[key], num_batch)
        data_dict['mean'] = mean_
        data_dict['std'] = std_
        return data_dict

    def get_list_id(self, list_id:list[int]):
        """get the list_id_train, list_id_val, list_id_test for  running stuff
        """
        list_id = sorted(list_id)
        list_id_train = list_id[:int(len(list_id)*0.5)]
        if self.is_multi:
            # only apply for multi_fx
            # choose the first 50% tasks for training, the last 50% is randomly chosen for validating and testing
            list_id = set(list_id)
            list_id_val = random.sample(sorted(list_id.difference(set(list_id_train))), len(list_id)//4)
            list_id_test = list(list_id.difference(set(list_id_train).union(list_id_val)))
        else:
            # bởi vì các dataset khác không multi
            # nên phải đảm bảo tính thứ tự trong huấn luyện
            list_id_val = list_id[int(len(list_id)*0.5):int(len(list_id)*0.75)]
            list_id_test = list_id[int(len(list_id)*0.75):]
        return list_id_train, list_id_val, list_id_test

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
                df = pd.read_csv(os.path.join(data_path, file_name)).to_numpy()[:,1:]

                # normalize data
                transformed_df, mean_, std_ = self._normalize_data(df, label, 0.2)

                # create label
                y = self._create_label(transformed_df, label)

                # concat transformed_df, y
                transformed_df = np.concatenate([transformed_df, y.reshape(-1,1)], axis=1)
                transformed_df = pd.DataFrame(transformed_df, columns=self.columns+['y'])
                transformed_df['unique_id'] = 0
                transformed_df['ds'] = pd.date_range(start='2000-01-01', periods=len(transformed_df), freq='h') # re-index all data
                return transformed_df, mean_, std_

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from common.constants import *

def create_label(df:np.ndarray, label:int, mode:str=CLF) -> np.ndarray:
    label = df[:,label]
    if mode == CLF:
        label = label[1:] - label[:-1]
        label = (label>0)*1.
        label = np.append([-1], label) # the first value is assigned to -1, it's removed anyway
    elif mode == REG:
        pass # if it is regression problem, we don't have to do anything
    return label

def normalize_data(df:np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df)

# create dataset with lookback window
def create_dataset(df:np.ndarray, look_back:int, label:int)->tuple[np.ndarray, np.ndarray]:
    size = df.shape[0] - look_back
    X = np.array([df[i : i + look_back] for i in range(size)])
    y = create_label(df, label)
    y = y[look_back:]
    return X, y.reshape(-1,1)

def get_data(look_back:int, label:int=-1, test_size:float=0.8, batch_size:int=32, data_dir:str=RAW_DATA_DIR) -> dict[str, dict[str, list[np.ndarray]]]:
    """
        read data from files and return a dict whose each value contains data from a file

        label (int, optional): The label to be predict the movement. Defaults to -1 (the final feature).
        batch_size (int, optional): If `batch_size=-1`, then we don't divide data into batches
    """
    cid = 0
    data_dict:dict[str, dict[str, pd.DataFrame]] = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            # read dataset, drop the datetime/index column
            df = pd.read_csv(os.path.join(data_dir, file_name)).to_numpy()[:,1:]

            # normalize data
            transformed_df = normalize_data(df)

            # create dataset with lookback window
            X, y = create_dataset(transformed_df, look_back, label)

            tmp_dict = {}
            tmp_dict['support_X'], tmp_dict['query_X'], tmp_dict['support_y'], tmp_dict['query_y'] = train_test_split(X, y, test_size=test_size, shuffle=False)
            if batch_size != -1: # do not divide data into batches
                for key in tmp_dict.keys():
                    num_batch = int(np.ceil(tmp_dict[key].shape[0]/batch_size))
                    tmp_dict[key] = np.array_split(tmp_dict[key], num_batch)
            data_dict[cid] = tmp_dict
            cid += 1

    return data_dict

def get_data_for_nhits(data_dir:str=RAW_DATA_DIR, label:int=-1):
    for file_name in os.listdir(data_dir):
        # data_dir should contain only 1 *.csv file
        if file_name.endswith('csv'):
            # read dataset, drop the datetime/index column
            df = pd.read_csv(os.path.join(data_dir, file_name))
            columns = list(df.columns[1:])
            df = df.to_numpy()[:,1:]

            # normalize data
            transformed_df = normalize_data(df)

            # create label
            y = create_label(transformed_df, label=label)

            # concat X, y
            transformed_df = np.concatenate([transformed_df, y.reshape(-1,1)], axis=1)
            transformed_df = pd.DataFrame(transformed_df, columns=columns+['y'])
            transformed_df['unique_id'] = 0
            transformed_df['ds'] = pd.date_range(start='2000-01-01', periods=len(transformed_df), freq='h')
            return transformed_df

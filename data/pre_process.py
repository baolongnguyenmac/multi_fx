import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

from common.constants import *

def create_label(df:np.ndarray) -> np.ndarray:
    label = df[1:,-1] - df[:-1,-1]
    label[label > 0] = 1
    label[label <= 0] = 0
    label = np.append([-1], label) # the first value is assigned to -1, it's removed anyway
    return label

def normalize_data(df:np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df)

# create dataset with lookback window
def create_dataset(df:np.ndarray, look_back:int)->tuple[np.ndarray, np.ndarray]:
    size = df.shape[0] - look_back
    X = np.array([df[i : i + look_back] for i in range(size)])
    y = create_label(df)
    y = y[look_back:]
    return X, y

def get_data(look_back:int, data_dir:str=RAW_DATA_DIR) -> dict[str, dict[str, np.ndarray]]:
    cid = 0
    data_dict:dict[str, dict[str, pd.DataFrame]] = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            # read dataset, drop the datetime column
            df = pd.read_csv(os.path.join(data_dir, file_name)).to_numpy()[:,1:]

            # normalize data
            transformed_df = normalize_data(df)

            # create dataset with lookback window
            X, y = create_dataset(transformed_df, look_back)

            tmp_dict = {}
            tmp_dict['support_X'], tmp_dict['query_X'], tmp_dict['support_y'], tmp_dict['query_y'] = train_test_split(X, y, test_size=0.8, shuffle=False)
            data_dict[cid] = tmp_dict
            cid += 1

    return data_dict

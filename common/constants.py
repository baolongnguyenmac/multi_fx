import os

# dirs
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/fx_data')
EXE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../execute')
PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pretrained')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../execute/log')

# dataset
MULTI_FX = 'multi_fx'
USD_JPY = 'USD_JPY'
ETT = 'ett'
ECL = 'ecl'
WTH = 'wth'
FACTORY_1 = 'factory_1'
FACTORY_2 = 'factory_2'

# mode
REG = 'regression'
CLF = 'classification'
VAL = 'Validation'
TEST = 'Test'

# based model
LSTM = 'lstm'
CNN = 'cnn'
LSTM_CNN = 'lstm_cnn'
ATT = 'attention'
VAL = 'validation'
TEST = 'test'

# layer
MAX_POOLING = 'max_pooling'
AVERAGE_POOLING = 'average_pooling'

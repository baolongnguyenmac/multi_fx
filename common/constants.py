import os

# dirs
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/fx_data')
EXE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../execute')
PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pretrained')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../execute/log')

# mode
REG = 'regression'
CLF = 'classification'
LSTM = 'lstm'
CNN = 'cnn'
LSTM_CNN = 'lstm+cnn'
ATT = 'attention'

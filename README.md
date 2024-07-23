# Multi FX

## Init data

- Crawl data from AV:
    - 60 currency pairs
    - About 2600 samples per pair (10 year, from 2014 to 2024)
    - A sample corresponds to a day

- Pre-process data:
    - Normalize data: z-score
    - Create dataset wrt. `look_back` window (`look_back: int`: we use `look_back` (historical) samples to predict the movement of foreign exchange)
    - Each pair is split into `support_set, query_set`

## Environment

- You should create a `*.sh` file to execute the following code:

```bash
# using python 3.11
conda create -n fx_env python=3.11
conda activate fx_env

# install machine learning libraries (use GPU version)
python3 -m pip install scikit-learn
python3 -m pip install tensorflow[and-cuda]
python3 -m pip install git+https://github.com/keras-team/keras-tuner.git
python3 -m pip install autokeras
pip3 install torch torchvision torchaudio

# install data-processed and visualization libraries
python3 -m pip install pandas
python3 -m pip install pandas-datareader
python3 -m pip install matplotlib
```

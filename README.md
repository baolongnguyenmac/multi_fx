# Multi FX

## [-] 2 ways to compute loss and accuracy

- For back-propagation, I use sum_loss, which is the total loss of model on all query-sets
- For evaluate as well as visualize, there are 2 ways to compute these metrics:
    [v] `macro-metrics` (implemented): compute loss of each task, then, compute the average mean of tasks. The pseudo-code is as follows:

        ```python
        task_accs = []
        task_losses = []
        sum_losses = 0.

        for task in batch_task:
            task_acc = []
            task_loss = []
            for batch_X, batch_y in task:
                pred = model(batch_X)
                loss = loss_fn(batch_y, pred)

                task_loss.append(loss)
                task_acc.append(acc_fn(batch_y, pred))

            print(mean([mean(batch_loss) for batch_loss in task_loss]))
            print(mean(task_acc))

            task_losses.append(mean([mean(batch_loss) for batch_loss in task_loss]))
            task_accs.append(mean(task_acc))
            sum_losses += sum([sum(batch_loss) for batch_loss in task_loss])

        batch_task_acc = mean(task_accs)
        batch_task_loss = mean(task_losses)
        ```

    [v] `micro-metrics` (implemented): compute loss of all batches (from all tasks) at once. **The outer_loss in the log file of `micro-metrics` is wrong**. The pseudo-code is as follows:

        ```python
        batch_acc = []
        batch_loss = []

        for task in batch_task:
            for batch_X, batch_y in task:
                pred = model(batch_X)
                loss = loss_fn(batch_y, pred)

                batch_loss.append(loss)
                batch_acc.append(acc_fn(batch_y, pred))

            print(mean([mean(batch_loss_) for batch_loss_ in batch_loss[-len(task:)]]))
            print(mean(batch_acc[-len(task):]))

        sum_loss = sum([sum(batch_loss_) for batch_loss_ in batch_loss]) # batch_loss contains many batch_loss_
        mean_loss = mean([mean(batch_loss_) for batch_loss_ in batch_loss])
        mean_acc = mean(batch_acc)
        ```

[-] I'm experimenting on both ways. `micro-metrics` is done and stored in `./pretrained_micro`. `macro-metrics` is just implemented. I will store in `./pretrained_macro`
    - I just experimented 20 models in macro-way, it's not bad. I will experiment more on this

## [-] Analyze

- I think I just found out something high in accuracy, I think it's would be better if I carefully analyze that `something`, reproduce them and try the old method ('cause I use new dataset)

### [-] Analyze `something`

[v] What if the label distribution of data are not uniform? $\to$ I will try other metrics such as `recall, precision, F1` $\to$ Don't have to use other metrics since their distributions are quite uniform (see `./img/dist_task.png`)

- What if data from `51-59` are easier than other? I think that there are 2 ways to check this hypothesis
    - Randomly choose dataset for training, validating, and testing (try with many random seed)
    - Run pre-trained models on data from `51-59`, if they all go well, then data from `51-59` are easy
    - In bad case, I can swap `val-set` and `test-set`, so I can obtain accuracy on test set smaller than train and val set

- Create an Excel file, I have to analyze these stuff. After all, it should obtain something like:
    - What is the best hyper-param?
    - What is the best batch-size of task?
    - What is the best batch-size of data?
    - What is the best number of rounds?
    - What is the potential model?
    - ... (fine-tune stuff)

- **What about LSTM+CNN?**

### - Try old method: `AutoKeras`

- Create a big data file that contains all data from pairs of currency
- Pre-process and use old models to run on it
- Check the result by employing the `*.log` files
    - Extract these kind of information from `*.log` files (`./img/metric_log.png`)
    - Then draw a learning line chart, which should show that the learning process is meaningless (the accuracies do not increase)
    - Conclusion: Reject `AutoKeras`

## [-] Problem of `based_model` & `meta_model`

- For now, `LSTM+CNN` is not working, `Attention` has not been implemented yet. **I think that I should implement the version of `LSTM+CNN` for classification first**
[v] In `meta_model`, `inner_opt` is initialized only once, I fix it

## [v] Init data

- Crawl data from AV:
    - 60 currency pairs
    - About 2600 samples per pair (10 year, from 2014 to 2024)
    - A sample corresponds to a day

- Pre-process data:
    - Normalize data: z-score
    - Create dataset wrt. `look_back` window (`look_back: int`: we use `look_back` (historical) samples to predict the movement of foreign exchange)
    - Each pair is split into `support_set, query_set`

## [v] Environment

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

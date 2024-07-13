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

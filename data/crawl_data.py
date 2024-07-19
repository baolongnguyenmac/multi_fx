from pandas_datareader.data import DataReader

def main():
    '''
    note:
        - av-forex-daily only allows to download data 25 times per day
        - we have to change the API to download more (in 1 day)
    '''

    list_pair = ['CHF/AUD', 'CHF/NZD', 'EUR/NZD', 'CAD/CHF', 'GBP/CHF', 'NOK/USD', 'CHF/JPY', 'EUR/JPY', 'USD/CHF', 'EUR/SEK', 'SGD/NOK', 'AUD/CAD', 'DKK/SEK', 'CAD/JPY', 'SEK/CAD', 'USD/MXN', 'TRY/JPY', 'NZD/CHF', 'NZD/USD', 'AUD/USD', 'SEK/USD', 'NZD/AUD', 'USD/ZAR', 'AUD/SGD', 'GBP/AUD', 'EUR/CAD', 'GBP/USD', 'GBP/CAD', 'EUR/CHF', 'USD/CAD', 'EUR/ISK', 'HKD/USD', 'SEK/NOK', 'USD/HKD', 'NZD/DKK', 'GBP/NZD', 'EUR/TRY', 'NZD/JPY', 'GBP/JPY', 'USD/JPY', 'AUD/DKK', 'EUR/AUD', 'USD/NOK', 'ISK/CHF', 'SEK/CHF', 'DKK/CHF', 'NZD/CAD', 'USD/TRY', 'AUD/CHF', 'NZD/SGD', 'USD/CNH', 'AUD/JPY', 'EUR/GBP', 'AUD/NZD', 'USD/SEK', 'CAD/USD', 'ZAR/JPY', 'EUR/NOK', 'DKK/ISK', 'EUR/USD']

    un_done = []

    for pair in list_pair:
        try:
            # get the API from here: https://www.alphavantage.co/support/#api-key
            # document for DataReader: https://pydata.github.io/pandas-datareader/readers/alphavantage.html?highlight=forex#module-pandas_datareader.av.forex
            df = DataReader(name=pair, data_source='av-forex-daily', start='2014-01-01', end='2024-12-31', api_key='36J3MISX9Q47K7NN')
            df.to_csv(f'./raw_data/{pair.replace('/', '_')}.csv')
        except:
            un_done.append(pair)

    print(f'Undone list: {un_done}')

if __name__ == '__main__':
    main()

'''
Long BackTesting Design
Given N indicators and Strategy A generate a buy, hold or sell signal.

Rules of BackTesting.

1. We have to pick an entry point.
2. We can ONLY sell what we have.
3. We can ONLY buy with what we have.
3. Every trade has a transaction fee.
Note: There are two types of fees
    There is a Taker Fee and a Maker Fee
    Taker Fee occurs when your order is instantly
    matched.
    Maker Fee occurs when your order is not instantly
    matched.

We are going to assume that we only buy or sell at the close
price so therefore there will be a taker fee cost.

In binance, there is a 0.1% spot trading fee and a
0.5% Instant Buy/Sell fee, we will be using the
instant buy/sell fee instead.
'''
import itertools as IT

import pandas as pd
import numpy as np

from random import random
from fastdtw import fastdtw
from pandas._libs.tslibs.offsets import BDay
from scipy.spatial.distance import euclidean
import time

import copy

from Indicators import SMAIndicator, EMAIndicator
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.signal import morlet
from scipy.spatial.distance import cosine

from numpy.fft import fft, ifft, rfft, irfft

import sys

'''
Useful for analysing the form of optimal signals. 
'''
def findMaxProfitK(price, k):
    # get the number of days `n`
    n = len(price)
    # base case
    if n <= 1:
        return 0
    # profit[i][j] stores the maximum profit gained by doing
    # at most `i` transactions till j'th day
    Signals = [[[[],[]] for x in range(n)] for y in range(k + 1)]
    Signals2 = [[{'From' : [], 'New' : []} for x in range(n)] for y in range(k + 1)]
    profit = [[0 for x in range(n + 1)] for y in range(k + 1)]
    # fill profit[][] in a bottom-up fashion

    for i in range(1, k + 1):
        # initialize `prev` diff to `-INFINITY`
        prev_diff = -sys.maxsize
        prev_xindex = 0
        prev_yindex = 0

        for j in range(1, n):
            # profit is 0 when
            # i = 0, i.e., for 0th day
            # j = 0, i.e., no transaction is being performed
            if i == 0 or j == 0:
                profit[i][j] = 0
            else:
                #prev_diff = max(prev_diff, profit[i - 1][j - 1] - price[j - 1])
                #print(f"Iteration = Transaction Number = {i}, Day Number = {j}")
                if prev_diff < profit[i-1][j-1] - price[j-1]: #Finding the max of - price[x] + profit[i - 1][x]
                    prev_diff = profit[i-1][j-1] - price[j-1] #Using previous iterations
                    #print(f"Buy = {price[j-1]} - Buy Index = {j-1}")
                    prev_xindex = i-1
                    prev_yindex = j-1
                    index_buy = j-1

                if profit[i][j-1] >= price[j] + prev_diff: #This is the same as price[j] - price[x] + profit[i-1][x]
                    profit[i][j] = profit[i][j-1]
                    Signals[i][j][0].extend(Signals[i][j-1][0])
                    Signals[i][j][1].extend(Signals[i][j-1][1])
                    Signals2[i][j]['From'] = [i, j-1]
                    Signals2[i][j]['New'] = []
                else:
                    profit[i][j] = price[j] + prev_diff
                    Signals[i][j][0].extend(Signals[prev_xindex][prev_yindex][0])
                    Signals[i][j][1].extend(Signals[prev_xindex][prev_yindex][1])

                    Signals2[i][j]['From'] = [prev_xindex, prev_yindex]
                    Signals2[i][j]['New'] = [index_buy, j]

                    Signals[i][j][0].append(index_buy)
                    Signals[i][j][1].append(j)

                    #print(f"Sell - Buy = {price[j]} - {price[index_buy]}  -- Sell Index  = {j}, Buy Index = {index_buy} ")
                #profit[i][j] = max(profit[i][j - 1], price[j] + prev_diff)

    return profit[k][n - 1], Signals[i][j][0], Signals[i][j][1], Signals, Signals2

def OptimisedfindMaxProfitK(price, k):
    # get the number of days `n`
    n = len(price)
    # base case
    if n <= 1:
        return 0
    # profit[i][j] stores the maximum profit gained by doing
    # at most `i` transactions till j'th day
    #Signals = [[[[],[]] for x in range(n)] for y in range(k + 1)]
    Signals2 = [[{'From' : [], 'New' : []} for x in range(n)] for y in range(k + 1)]
    profit = [[0 for x in range(n + 1)] for y in range(k + 1)]
    # fill profit[][] in a bottom-up fashion

    for i in range(1, k + 1):
        # initialize `prev` diff to `-INFINITY`
        prev_diff = -sys.maxsize
        prev_xindex = 0
        prev_yindex = 0

        for j in range(1, n):
            # profit is 0 when
            # i = 0, i.e., for 0th day
            # j = 0, i.e., no transaction is being performed
            if i == 0 or j == 0:
                profit[i][j] = 0
            else:
                #prev_diff = max(prev_diff, profit[i - 1][j - 1] - price[j - 1])
                #print(f"Iteration = Transaction Number = {i}, Day Number = {j}")
                if prev_diff < profit[i-1][j-1] - price[j-1]: #Finding the max of - price[x] + profit[i - 1][x]
                    prev_diff = profit[i-1][j-1] - price[j-1] #Using previous iterations
                    prev_xindex = i-1
                    prev_yindex = j-1
                    index_buy = j-1

                if profit[i][j-1] >= price[j] + prev_diff: #This is the same as price[j] - price[x] + profit[i-1][x]
                    profit[i][j] = profit[i][j-1]
                    Signals2[i][j]['From'] = [i, j-1]
                    Signals2[i][j]['New'] = []
                else:
                    profit[i][j] = price[j] + prev_diff
                    Signals2[i][j]['From'] = [prev_xindex, prev_yindex]
                    Signals2[i][j]['New'] = [index_buy, j]

    xind = k
    yind = n-1
    BuySignals = []
    SellSignals = []

    while True:
        S = Signals2[xind][yind]['New']
        if len(S) != 0:
            BuySignals.append(S[0])
            SellSignals.append(S[1])

        index = Signals2[xind][yind]['From']
        if len(index) == 0:
            break
        xind = index[0]
        yind = index[1]

    indicator = np.ones(len(price))
    indicator *= -1
    #-1 is hold
    # 1 is buy
    # 0 is sell
    for i,j in zip(BuySignals, SellSignals):
        indicator[i] = 1
        indicator[j] = 0

    return profit[k][n - 1], BuySignals, SellSignals, indicator

#Brute Force Version
def BrutefindMaxProfit(price, k):
    # get the number of days `n`
    n = len(price)
    # base case
    if n <= 1:
        return 0

    # profit[i][j] stores the maximum profit gained by doing
    # at most `i` transactions till j'th day
    profit = [[0 for x in range(n)] for y in range(k + 1)]

    Signals = [[[[],[]] for x in range(n)] for y in range(k + 1)]

    # fill profit[][] in a bottom-up fashion
    for i in range(k + 1):
        for j in range(n):
            # profit is 0 when
            # i = 0, i.e., for 0th day
            # j = 0, i.e., no transaction is being performed

            if i == 0 or j == 0:
                profit[i][j] = 0
            else:
                max_so_far = 0
                index_buy = 0
                index_sell = 0
                for x in range(j):
                    #profit[i-1][x] is the maximum profit, up to i-1 transactions at index x, which is less than j.
                    curr_price = price[j] - price[x] + profit[i - 1][x]

                    # When we do price[j] - price[x], we are finding the profit between 2 points.
                    # When we add profit[i-1][x] we are adding, the profits up to i-1 transactions UNTIL point X.
                    # It is NOT overlapping
                    if max_so_far < curr_price:
                        #print(f"Buy at {price[j]} - {price[i]}")
                        prev_xindex = i-1
                        prev_yindex = x
                        index_buy = x
                        index_sell = j
                        max_so_far = curr_price
                # We then compare if the maximum amount found is greater than performing at most i transactions
                # until J-1 points.
                if profit[i][j-1] >= max_so_far:
                    profit[i][j] = profit[i][j-1]
                    Signals[i][j][0].extend(Signals[i][j-1][0])
                    Signals[i][j][1].extend(Signals[i][j-1][1])

                else:
                    profit[i][j] = max_so_far
                    Signals[i][j][0].extend(Signals[prev_xindex][prev_yindex][0])
                    Signals[i][j][1].extend(Signals[prev_xindex][prev_yindex][1])
                    Signals[i][j][0].append(index_buy)
                    Signals[i][j][1].append(index_sell)

    return profit[k][n - 1], Signals[k][n-1][0], Signals[k][n-1][1]

#Literally finds the best Buy / Sell without a restriction on k transactions
def findMaxProfit(price):
    # keep track of the maximum profit gained
    #buyprices = [element * 1 / (1 - r) for element in price]
    #sellprices = [element * (1 - r) for element in price]
    profit = 0
    # initialize the local minimum to the first element's index
    j = 0
    Signals = np.full(len(price), -1)
    # start from the second element
    for i in range(1, len(price)):
        # update the local minimum if a decreasing sequence is found
        if price[i - 1] > price[i]:
            j = i
        # sell shares if the current element is the peak, i.e.,
        # (`previous <= current > next`)
        if price[i - 1] <= price[i] and \
                (i + 1 == len(price) or price[i] > price[i + 1]):
            profit += (price[i] - price[j])
            Signals[j] = 0
            Signals[i] = 1

    return Signals

def getDailyCrypto():
    CryptoData = []
    Variables = ["BNBUSDT", "ADAUSDT", "BTCUSDT", "DASHUSDT",
                 "DASHUSDT", "EOSUSDT", "ETCUSDT", "ETHUSDT",
                 "LINKUSDT", "LTCUSDT", "NEOUSDT", "QTUMUSDT",
                 "TRXUSDT", "XLMUSDT", "XRPUSDT", "ZECUSDT"]

    for x in Variables:
        path = 'CryptoMinuteData/Binance_' + x + "_d.csv"
        data = pd.read_csv(path)
        CryptoData.append(data)

    return CryptoData, Variables

# Use this Gauss_filter instead
def Gauss_Filter(Input, Sigma):
    lw = int(4 * Sigma + 0.5)
    w = np.arange(-lw, lw + 1)
    phi = np.exp(-1 / (2 * Sigma ** 2) * (w ** 2))
    phi /= phi.sum()
    y = np.convolve(Input, phi, mode='')
    return y

def getData():
    CryptoData = []
    Variables = ["BNBUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT",
                 "NEOUSDT", "QTUMUSDT", "XRPUSDT", "LTCUSDT"]

    for x in Variables:
        path = 'CryptoMinuteData/Binance_' + x + "_1h.csv"
        data = pd.read_csv(path)
        CryptoData.append(data)

    return CryptoData, Variables

def DowJones():
    path = 'Dow_Jones_1896.csv'
    df = pd.read_csv(path)
    df.Date = pd.to_datetime(df.Date)
    df = df[["Date", "Close"]]
    df.rename(columns={'Close': 'close'}, inplace=True)
    isBusinessDay = BDay().is_on_offset
    match_series = pd.to_datetime(df['Date']).map(isBusinessDay)
    df = df[match_series]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

'''
In this simulation we have to follow every single decision made, this simulation has no memory of previous buy and sell
prices BUT only what there last action was (whether they bought or sold), and how much money they have. 
We ONLY go LONG in this simulation AND we THERE ARE NO TRANSACTION FEES.
'''
def BackTestGoingLongNaive(data):
    buySignal, sellSignal, transactionFee, initial, amount, shares, index, size = 1, 0, 0.005, 10000, 10000, 0, 1000, len(
        data)
    x = 0
    prevAmount = 0
    sells = 0
    Profits = []

    BuyIndex = []
    SellIndex = []
    BuyPrice = []
    SellPrice = []
    count = 0

    Dates = []
    ShareTransfers = []
    Actions = []
    ActionsStrings = []
    PriceTransfers = []
    Price = []

    # Where start from is important however.
    while index < size:
        x = GaussianSimpleMovingAverageStrategy(data['close'][count:index])
        # print(len(data['close'][count:index]))
        # x = SimpleMovingAverageStrategy(data['close'][:index])
        if x == 0:
            # print(f"Date : {data['date'][index]}")
            count += 1
            index += 1
            continue
        elif x == 1:
            if buySignal:
                shares = (amount - (amount * transactionFee)) / (data['close'][index])
                Dates.append(data['date'][index])
                ShareTransfers.append(shares)
                Actions.append(1)
                PriceTransfers.append(amount - (amount * transactionFee))
                Price.append(data['close'][index])
                ActionsStrings.append("Buy")
                # print(f"Date : {data['date'][index]}")
                print(
                    f"Shares bought = {shares}, at price = {data['close'][index]}, money spent = {amount}, actual money spent = {amount - (amount * transactionFee)}")
                prevAmount = amount
                amount = 0
                buySignal = 0
                sellSignal = 1
                BuyIndex.append(index)
                BuyPrice.append(data['close'][index])

        elif x == -1:
            if sellSignal:
                amount = (shares * data['close'][index])
                Dates.append(data['date'][index])
                ShareTransfers.append(shares)
                Actions.append(0)
                PriceTransfers.append(amount - (amount * transactionFee))
                Price.append(data['close'][index])
                ActionsStrings.append("Sell")
                # print(f"Date : {data['date'][index]}")
                print(
                    f"Shares sold = {shares} at price = {data['close'][index]}, money recieved = {amount}, actual money recieved = {amount - amount * transactionFee}")
                amount = amount - amount * transactionFee
                Profits.append(amount - prevAmount)
                shares = 0
                sells += 1
                sellSignal = 0
                buySignal = 1
                SellIndex.append(index)
                SellPrice.append(data['close'][index])
        # print(f"Date : {data['date'][index]}")
        index += 1
        count += 1
    # print(size)
    dic = dict()
    dic['Date'] = Dates
    dic['Share Transfers'] = ShareTransfers
    dic['Price Transfers'] = PriceTransfers
    dic['Price'] = Price
    dic['Action'] = ActionsStrings
    dic['Buy/Sell'] = Actions
    df = pd.DataFrame(dic)

    if sellSignal:
        return prevAmount / initial, BuyIndex, SellIndex, BuyPrice, SellPrice, df
    else:
        return amount / initial, BuyIndex, SellIndex, BuyPrice, SellPrice, df
'''
There is 1-alpha probability that the trade execution will be completed. 
There is an alpha probability that it will not be executed.

Theory: If there is a large frequency of signals, there should be a
greater chance that these signals should be acted upon.
'''
def BackTestGoingLongRandomised(data, buyProb, sellProb):
    buySignal, sellSignal, transactionFee, initial, amount, shares, index, size = 1, 0, 0.005, 10000, 10000, 0, 1000, len(
        data)
    SellFrequency = 0
    BuyIndex = []
    BuyPrice = []
    SellIndex = []
    SellPrice = []
    # Where start from is important however.
    while index < size:
        x = SimpleMovingAverageStrategy(data['close'][:index])
        prob = np.random.uniform(0, 1, 1)  # Decides if we should even consider a decision
        if x == 0:
            index += 1
            continue
        elif x == 1:
            if buySignal and prob <= buyProb:  # Higher probability that we will go through with this decision, we are more confident in buying.
                shares = (amount - (amount * transactionFee)) / (data['close'][index])
                amount = 0
                buySignal = 0
                sellSignal = 1
                BuyIndex.append(index)
                BuyPrice.append(data['close'][index])
        elif x == -1:
            if sellSignal and prob <= sellProb:  # False Signals, may be generated.
                amount = (shares * data['close'][index])
                # print(f"Shares sold = {shares} at price = {data['close'][index]}, money recieved = {amount}, actual money recieved = {amount - amount * transactionFee}")
                amount = amount - amount * transactionFee
                SellFrequency += 1
                # print(f"Profit Made : {SellProfits}")
                shares = 0
                sellSignal = 0
                buySignal = 1
                SellIndex.append(index)
                SellPrice.append(data['close'][index])

        index += 1
    if sellSignal:
        amount = (shares * data['close'][index - 1])
        amount = amount - amount * transactionFee
        return amount / initial, BuyIndex, SellIndex, BuyPrice, SellPrice
    # else:
    # print(f"Number of Trades: {SellFrequency}, Returns: {amount/initial}")
    # print(amount/initial)
    return amount / initial, BuyIndex, SellIndex, BuyPrice, SellPrice

def BackTestStrategyA(data):
    buySignal, sellSignal, transactionFee, initial, amount, shares, index, size = 1, 0, 0.005, 10000, 10000, 0, 1, len(
        data)
    x = 0
    prevAmount = 0
    sells = 0
    Profits = []

    BuyIndex = []
    SellIndex = []
    BuyPrice = []
    SellPrice = []

    # Where start from is important however.
    data = InitialiseStrategyA(data)

    while index < size:
        x = ExecuteStrategyA(data[:index])
        if x == 0:
            index += 1
            continue
        elif x == 1:
            if buySignal:
                shares = (amount - (amount * transactionFee)) / (data['close'][index])
                # print(f"Shares bought = {shares}, at price = {data['close'][index]}, money spent = {amount},actual money spent = {amount - (amount*transactionFee)}")
                prevAmount = amount
                amount = 0
                buySignal = 0
                sellSignal = 1
                BuyIndex.append(index)
                BuyPrice.append(data['close'][index])

        elif x == -1:
            if sellSignal:
                amount = (shares * data['close'][index])
                # print(f"Shares sold = {shares} at price = {data['close'][index]}, money recieved = {amount}, actual money recieved = {amount - amount * transactionFee}")
                amount = amount - amount * transactionFee
                Profits.append(amount - prevAmount)
                shares = 0
                sells += 1
                sellSignal = 0
                buySignal = 1
                SellIndex.append(index)
                SellPrice.append(data['close'][index])

        index += 1
    # print(size)
    if sellSignal:
        return prevAmount / initial, BuyIndex, SellIndex, BuyPrice, SellPrice
    else:
        return amount / initial, BuyIndex, SellIndex, BuyPrice, SellPrice
    return
# This just refers to whether we should buy or sell purely based on data.

'''
Using Gaussian Cross overs for signals.
'''
def GaussianSimpleMovingAverageStrategy(data):
    SmallWindow = 9 * 24
    BigWindow = 24 * 24
    Var = np.var(data)
    GaussData = Gauss_Filter(data, Var)
    # print(GaussData)
    # GaussData = data
    # gaussData = pd.Series(gaussian_filter1d())
    DataShort = data[len(GaussData) - 1 - SmallWindow:]
    DataLong = data[len(GaussData) - 1 - BigWindow:]

    SMA_9 = SMAIndicator(DataShort, window=SmallWindow, fillna=False)
    SMA_25 = SMAIndicator(DataLong, window=BigWindow, fillna=False)
    SMA_9 = SMA_9.dropna()
    SMA_25 = SMA_25.dropna()

    SMA_9 = SMA_9.reset_index(drop=True)
    SMA_25 = SMA_25.reset_index(drop=True)
    before = 0
    current = 1

    if SMA_9[before] > SMA_25[before] and SMA_9[current] < SMA_25[current]:
        return -1
    elif SMA_9[before] < SMA_25[before] and SMA_9[current] > SMA_25[current]:
        return 1
    return 0

def SimpleMovingAverageStrategy(data):
    # We need data from (n - 1) - w, where w is window size.
    # This is because we ONLY need the current and the data before the current for this.
    SmallWindow = 9 * 24
    BigWindow = 24 * 24

    DataShort = data[len(data) - 1 - SmallWindow:]
    DataLong = data[len(data) - 1 - BigWindow:]

    SMA_9 = SMAIndicator(DataShort, window=SmallWindow, fillna=False)
    SMA_25 = SMAIndicator(DataLong, window=BigWindow, fillna=False)

    SMA_9 = SMA_9.dropna()
    SMA_25 = SMA_25.dropna()

    before = len(data) - 2
    current = len(data) - 1

    if SMA_9[before] > SMA_25[before] and SMA_9[current] < SMA_25[current]:
        return -1
    elif SMA_9[before] < SMA_25[before] and SMA_9[current] > SMA_25[current]:
        return 1
    return 0

# Using the data we have so far predict what the next value would be.
def DummyStrategy(data):
    x = random()
    if x < 0.33:
        return -1  # Sell
    elif x < 0.66:
        return 1  # Buy
    elif x < 1:
        return 0  # Hold
    return random() < 0.5

def InitialiseStrategyA(df):
    SmallWindow = 9 * 24
    BigWindow = 24 * 24
    GiantWindow = 4000
    df['Forecast'] = df['close'].rolling(window=GiantWindow).mean()
    df['MA_SHORT'] = df['close'].rolling(window=SmallWindow).mean()
    df['MA_LONG'] = df['close'].rolling(window=BigWindow).mean()
    df.dropna()

    return df

def ExecuteStrategyA(data):
    before = len(data) - 2
    current = len(data) - 1
    if data['Forecast'][current] < data['close'][current]:
        # print(f"{data['MA_LONG'][before]} - {data['MA_SHORT'][before]}")
        if data['MA_LONG'][before] > data['MA_SHORT'][before] and data['MA_LONG'][current] < data['MA_SHORT'][current]:
            return 1
        elif data['MA_LONG'][before] < data['MA_SHORT'][before] and data['MA_LONG'][current] > data['MA_SHORT'][
            current]:
            return -1

    return 0

def Convolve(x, y):
    x_f = fft(x)
    y_f = fft(y)
    z_f = x_f * y_f
    print(ifft(z_f))
    return ifft(z_f).real

def FrequencyDomainConvolutions(df):
    # Given Data Find the price changes.
    # Moving Gauissian M
    df['Gaussian'] = gaussian_filter(df['close'], sigma=50)
    df['Gaussian1'] = gaussian_filter(df['close'], sigma=100)
    df['Gaussian2'] = gaussian_filter(df['close'], sigma=20)

    fig = go.Figure([go.Scatter(x=df.index, y=df['Gaussian'])])
    # fig.add_trace(go.Scatter(x=df.index, y=df.close))
    # fig.add_trace(go.Scatter(x=df.index, y=df['Gaussian1']))
    fig.add_trace(go.Scatter(x=df.index, y=df['Gaussian2']))

    fig.update_xaxes(rangeslider_visible=True)
    fig.data = fig.data[::-1]

    fig.show()
    return

def reset_my_index(df):
    res = df[::-1].reset_index(drop=True)
    return (res)

def AllSubPlots(CryptoData, Variables):
    fig = make_subplots(rows=2, cols=2)
    row = 1
    cols = 1
    for i in range(0, 4):
        df = CryptoData[i]
        df = reset_my_index(df)
        fig.add_trace(go.Scatter(x=df.index, y=df.close, name=Variables[i]), row=row, col=cols)
        cols += 1
        if cols == 3:
            cols = 1
            row = 2
    fig.update_layout(height=1200, width=1200, title_text="Coin Data")
    plotly.offline.plot(fig, filename="SubPlots.html")

    fig1 = make_subplots(rows=2, cols=2)
    row = 4
    cols = 4
    for i in range(4, 8):
        df = CryptoData[i]
        df = reset_my_index(df)
        fig1.add_trace(go.Scatter(x=df.index, y=df.close, name=Variables[i]), row=row - 3, col=cols - 3)
        cols += 1
        if cols == 6:
            cols = 4
            row = 5
    fig1.update_layout(height=1200, width=1200, title_text="Coin Data")
    plotly.offline.plot(fig1, filename="SubPlots2.html")
    return

'''
Important Graphing used to actually graph the buy and sell signals.
'''
def PlotSignalGraph(df, x, code):
    fig = go.Figure([go.Scatter(x=df.index, y=df['close'])])
    fig.update_xaxes(rangeslider_visible=True)
    fig.add_trace(go.Scatter(x=x[1], y=x[3], mode='markers', name='Buy', marker_color='green', marker_symbol=5,
                             marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=x[2], y=x[4], mode='markers', name='Sell', marker_color='red', marker_symbol=6,
                             marker=dict(size=10)))
    fig.update_layout(title=code + " Trade Signals, S = 216, L = 576")
    code = code + '.html'
    plotly.offline.plot(fig, filename=code)
    fig.show()
    return

def MovingAverageGraphWithSignals(df, code, x, a, b, title):
    df['Forecast'] = df['close'].rolling(window=4000).mean()
    df['Long'] = df['close'].rolling(window=9 * a).mean()
    df['Short'] = df['close'].rolling(window=24 * b).mean()
    fig = go.Figure([go.Scatter(x=df.index, y=df['close'], name="Price Data", line=dict(color='purple', width=0.75))])
    fig.update_xaxes(rangeslider_visible=True)
    fig.add_trace(go.Scatter(x=x[1], y=x[3], mode='markers', name='Buy', marker_color='green', marker_symbol=5,
                             marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=x[2], y=x[4], mode='markers', name='Sell', marker_color='red', marker_symbol=6,
                             marker=dict(size=10)))
    g = 9 * a
    h = 24 * b

    str1 = "SMA-" + str(g)
    str2 = "SMA-" + str(h)

    fig.add_trace(go.Scatter(x=df.index, y=df.Short, name=str1, line=dict(color='orange', width=0.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df.Long, name=str2, line=dict(color='black', width=0.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df.Forecast, name="SMA-4000", line=dict(color='blue', width=0.5)))
    code = code + 'MovingAverageWithSignals.html'

    fig.data = fig.data[::-1]
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple"
        )
    )

    plotly.offline.plot(fig, filename=code)
    return

def MovingAverageGraph(df, code):
    df['MA_3000'] = df['close'].rolling(window=3000).mean()
    df['MA_216'] = df['close'].rolling(window=216).mean()
    df['MA_576'] = df['close'].rolling(window=576).mean()
    fig = go.Figure([go.Scatter(x=df['date'], y=df['close'], name="Price Data")])
    fig.update_xaxes(rangeslider_visible=True)
    fig.add_trace(go.Scatter(x=df.date, y=df.MA_216, name="SMA-216", line=dict(color='orange', width=0.5)))
    fig.add_trace(go.Scatter(x=df.date, y=df.MA_576, name="SMA-576", line=dict(color='green', width=0.5)))
    fig.add_trace(go.Scatter(x=df.date, y=df.MA_3000, name="SMA-3000", line=dict(color='red', width=0.5)))
    code = code + 'MovingAverage.html'

    fig.data = fig.data[::-1]
    plotly.offline.plot(fig, filename=code)
    return

def MinLength(CryptoData):
    l = []
    for i in CryptoData:
        print(len(i))
        l.append(len(i))

    return min(l)

def DataTransform1(df):
    df = reset_my_index(df)
    df['target'] = df['close']
    return df

def GaussTransform(df, l):
    df = df.head(l)
    df = DataTransform1(df)
    df['target'] = gaussian_filter(df['close'], sigma=15)
    return df

def NormalizeGaussTransform(df, l):
    df = df.head(l)
    df = DataTransform1(df)
    df['Gauss'] = gaussian_filter(df['close'], sigma=15)
    df['target'] = (df.Gauss - df.Gauss.min()) / (df.Gauss.max() - df.Gauss.min())
    return df

def SubPlotTemplate(CryptoData, Variables, filename1, filename2):
    minL = MinLength(CryptoData)
    fig = make_subplots(rows=2, cols=2)
    row = 1
    cols = 1
    for i in range(0, 4):
        df = CryptoData[i]
        df = NormalizeGaussTransform(df, minL)
        fig.add_trace(go.Scatter(x=df.index, y=df.target, name=Variables[i]), row=row, col=cols)
        cols += 1
        if cols == 3:
            cols = 1
            row = 2
    fig.update_layout(height=1200, width=1200, title_text="Coin Data")
    plotly.offline.plot(fig, filename=filename1)

    fig1 = make_subplots(rows=2, cols=2)
    row = 4
    cols = 4
    for i in range(4, 8):
        df = CryptoData[i]
        df = NormalizeGaussTransform(df, minL)
        fig1.add_trace(go.Scatter(x=df.index, y=df.target, name=Variables[i]), row=row - 3, col=cols - 3)
        cols += 1
        if cols == 6:
            cols = 4
            row = 5
    fig1.update_layout(height=1200, width=1200, title_text="Coin Data")
    plotly.offline.plot(fig1, filename=filename2)
    return

def RMSE(CryptoData, Variables):
    TimeSeriesData = {}
    RMSEMatrix = np.zeros((8, 8))
    Rankings = {}
    labels = []
    minL = MinLength(CryptoData)
    for i in range(0, len(CryptoData)):
        df = CryptoData[i]
        df = NormalizeGaussTransform(df, minL)
        TimeSeriesData[i] = df.target
        Rankings[Variables[i]] = i
        labels.append(Variables[i])

    # How to calculate the
    for i in range(0, 8):
        for j in range(0, 8):
            RMSEMatrix[i][j] = mean_squared_error(TimeSeriesData[i], TimeSeriesData[j], squared=False)

    count = 0
    for i in RMSEMatrix:
        args = np.argsort(i)
        PrintOutInOrder(Variables[count], args, labels)
        count += 1

def CosineSimilarity(CryptoData, Variables):
    TimeSeriesData = {}
    SimMatrix = np.zeros((8, 8))
    Rankings = {}
    labels = []
    minL = MinLength(CryptoData)
    for i in range(0, len(CryptoData)):
        df = CryptoData[i]
        df = NormalizeGaussTransform(df, minL)
        TimeSeriesData[i] = df.target
        Rankings[Variables[i]] = i
        labels.append(Variables[i])
    # How to calculate the
    for i in range(0, 8):
        for j in range(0, 8):
            SimMatrix[i][j] = 1 - cosine(TimeSeriesData[i], TimeSeriesData[j])

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    df = pd.DataFrame(SimMatrix, columns=labels, index=labels)
    print(df)

    count = 0
    for i in SimMatrix:
        args = np.flip(np.argsort(i))
        PrintOutInOrder(Variables[count], args, labels)
        count += 1

def PrintOutInOrder(str, args, lists):
    print(f"For {str} :", end=" ")
    count = 0
    for i in args:
        if count == 0:
            count += 1
            continue
        print(lists[i], end=" ")
        count += 1

    print()

def DynamicTimeWarping(CryptoData, Variables):
    TimeSeriesData = {}
    DTWMatrix = np.zeros((8, 8))
    Rankings = {}
    labels = []
    minL = MinLength(CryptoData)
    for i in range(0, len(CryptoData)):
        df = CryptoData[i]
        df = NormalizeGaussTransform(df, minL)
        TimeSeriesData[i] = df.target
        Rankings[Variables[i]] = i
        labels.append(Variables[i])
    # How to calculate the
    for i in range(0, 8):
        for j in range(0, 8):
            distance, path = fastdtw(TimeSeriesData[i], TimeSeriesData[j], dist=euclidean)
            DTWMatrix[i][j] = distance

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    df = pd.DataFrame(DTWMatrix, columns=labels, index=labels)
    print(df)

    count = 0
    for i in DTWMatrix:
        args = np.argsort(i)
        PrintOutInOrder(Variables[count], args, labels)
        count += 1
    return

##Need to use / write about other indicators as well !!
if __name__ == "__main__":


    CryptoData, Variables = getData()
    DowData = DowJones()
    x = reset_my_index(CryptoData[5])
    z = OptimisedfindMaxProfitK(x['close'], 1000)

    Price = x['close']
    Volume = x['Volume' + Variables[5]]
    TradeCount = x['tradecount']

    n = len(x['close'])

    OptimalSignals = z[3]
    k = 24*14
    #Feature Creation
    PriceFeatureMatrix = np.zeros(shape=(k,n-k))
    VolumeFeatureMatrix = np.zeros(shape=(k, n-k))
    TradeCountFeatureMatrix = np.zeros(shape=(k, n-k))
    i = 0
    for i in range(0, n-k):
        PriceFeatureMatrix[i] = Price[i:k+i]
        VolumeFeatureMatrix[i] = Volume[i:k+i]
        TradeCountFeatureMatrix[i] = TradeCount[i:k+i]

    x['close'] = x['tradecount']/x['close']

    #Note the 335th - Moment will be
    #There will be n-k vectors.


    #x = DowData
    '''

    BuyList =  [13, 481, 845, 2272, 2803, 3155, 3453, 3900, 4148, 4282, 4616, 4619, 4654, 4686, 4725, 4775, 4865, 4923, 4994,
      5052, 5163, 5191, 5347, 5403, 5550, 5592, 5635, 5649, 5677, 5731, 5786, 5812, 5843, 5874, 5904, 5936, 5957, 5961,
      5967, 5972, 6019, 6038, 6055, 6094, 6128, 6148, 6175, 6182, 6201, 6214, 6221, 6226, 6238, 6244, 6246, 6249, 6252,
      6259, 6262, 6268, 6270, 6291, 6307, 6312, 6330, 6335, 6341, 6352, 6376, 6394, 6454, 6487, 6531, 6544, 6589, 6604,
      6612, 6617, 6643, 6655, 6659, 6665, 6668, 6683, 6685, 6689, 6701, 6706, 6713, 6718, 6721, 6728, 6731, 6735, 6747,
      6753, 6759, 6768, 6771, 6777, 6799, 6804, 6815, 6823, 6835, 6849, 6863, 6877, 6897, 6904, 6918, 6923, 6939, 6961,
      6973, 6976, 6991, 7017, 7027, 7040, 7061, 7095, 7104, 7112, 7130, 7157, 7176, 7186, 7206, 7214, 7227, 7241, 7280,
      7378, 7443, 7454, 7529, 7729, 7778, 7808, 7869, 7973, 8014, 8173, 8461, 8607, 8749, 8805, 8895, 8911, 8957, 8987,
      9003, 9043, 9064, 9105, 9111, 9128, 9140, 9164, 9239, 9305, 9348, 9383, 9429, 9442, 9465, 9513, 9564, 9588, 9623,
      9628, 9640, 9677, 9695, 9730, 9761, 9767, 9945, 9969, 10029, 10072, 10136, 10231, 10252, 10258, 10269, 10311,
      10376, 10456, 10479, 10587, 10708, 10753, 10830, 10893, 10912, 10918, 10930, 10947]

    SellList =  [459, 505, 992, 2779, 2911, 3351, 3867, 4092, 4196, 4614, 4617, 4632, 4659, 4698, 4739, 4839, 4871, 4975, 5011,
      5147, 5178, 5260, 5370, 5505, 5576, 5626, 5645, 5663, 5687, 5755, 5809, 5818, 5864, 5899, 5933, 5948, 5960, 5962,
      5968, 5993, 6034, 6041, 6072, 6118, 6137, 6163, 6178, 6194, 6206, 6217, 6225, 6236, 6240, 6245, 6247, 6251, 6254,
      6260, 6266, 6269, 6275, 6295, 6309, 6320, 6331, 6339, 6351, 6358, 6389, 6445, 6481, 6515, 6541, 6576, 6599, 6609,
      6615, 6640, 6652, 6658, 6663, 6667, 6670, 6684, 6686, 6693, 6703, 6711, 6714, 6719, 6723, 6730, 6732, 6739, 6748,
      6755, 6766, 6770, 6775, 6780, 6802, 6811, 6818, 6830, 6840, 6852, 6867, 6887, 6900, 6912, 6921, 6936, 6948, 6965,
      6975, 6986, 6997, 7020, 7031, 7045, 7093, 7096, 7108, 7124, 7154, 7164, 7178, 7191, 7212, 7224, 7233, 7261, 7318,
      7407, 7449, 7477, 7604, 7739, 7801, 7839, 7959, 7980, 8074, 8251, 8598, 8714, 8766, 8880, 8899, 8917, 8983, 8991,
      9040, 9044, 9083, 9107, 9113, 9134, 9145, 9200, 9262, 9321, 9369, 9394, 9436, 9454, 9502, 9561, 9584, 9593, 9625,
      9635, 9666, 9685, 9715, 9746, 9766, 9823, 9956, 10016, 10039, 10091, 10229, 10234, 10254, 10264, 10274, 10320,
      10398, 10471, 10506, 10673, 10729, 10791, 10875, 10906, 10915, 10925, 10935, 10954]
    '''

    BuyList = z[1]
    SellList = z[2]
    
    BuyPrices = []
    for i in BuyList:
        BuyPrices.append(x['close'][i])

    print(z[0]/BuyPrices[0])

    SellPrices = []
    for i in SellList:
        SellPrices.append(x['close'][i])

    fig = go.Figure([go.Scatter(x=x.index, y=x['close'])])
    fig.update_xaxes(rangeslider_visible=True)
    fig.add_trace(go.Scatter(x=BuyList, y=BuyPrices, mode='markers', name='Buy', marker_color='green', marker_symbol=5,
                             marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=SellList, y=SellPrices, mode='markers', name='Sell', marker_color='red', marker_symbol=6,
                             marker=dict(size=10)))
    fig.update_layout(title=Variables[5] + "  Optimal Buy and Sell Signals for 1000 trades")
    code = Variables[5] + 'Optimal' + '.html'
    plotly.offline.plot(fig, filename=code)
    fig.show()



    #print(testlist)
    # for i in range (0, len(CryptoData)):
    #    x = BackTestGoingLongNaive(reset_my_index(CryptoData[i]))
    #    PlotSignalGraph(reset_my_index(CryptoData[i]), x, Variables[i])
    #    filename = Variables[i] + ".csv"
    #    x[5].to_csv(filename)

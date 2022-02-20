import numpy as np
import pandas as pd
from ta import momentum, volume, volatility, trend

def SMA(data, window):
    sma = data.rolling(window = window).mean()
    return sma

#Bollinger Bands
def BollingerBandsp2(data, sma, window):
    std = data.rolling(window=window).std()
    upper_bound = sma + std * 2
    lower_bound = sma - std * 2
    return upper_bound, lower_bound

def EMA(data, window, smoothing=2):
    ema = [sum(data[:window])/window]
    for price in data[window:]:
        ema.append((price * (smoothing/(1+window))) + ema[-1] *(1- (smoothing/(1+window))))
    
    empty = np.empty(len(data) - len(ema))
    ema = np.concatenate((ema,empty))
    return ema


#Used as breakout indicators for weekly/monthly time frames
#Use in shorter time frames to define likely support and resistence
#Levels at the lower and upper acceleration bands.
def ABANDS(High, Low, Window):
    Upper_Band = SMA(High*(1+4*(High-Low)/(High+Low)),Window)
    Lower_Band = SMA(Low * (1+4*(High-Low)/(High+Low)),Window)
    return Upper_Band, Lower_Band

# Cumulatice indicator that uses volume and price to assess
# whether a stock is being accumulated or distributed
# Seeks to identify divergences between the stock price and the volume flow
def AccumulationDistribution(high,close,low,volume):
    MFM = ((close - low) - (high - close))/(high-low)
    MFMV = MFM * volume
    CMFMV = np.cumsum(MFMV)
    return CMFMV


'''
Momentum Indicators
'''
def ROCIndicator(close,window):
    return momentum.ROCIndicator(close,window).roc()

def RSIIndicator(close,window): 
    return momentum.RSIIndicator(close,window).rsi()

def StochRSIIndicator(close, window, smooth1, smooth2):
    return momentum.StochRSIIndicator(close,window,smooth1,smooth2).stochrsi(), momentum.StochRSIIndicator(close,window,smooth1,smooth2).stochrsi_k(),momentum.StochRSIIndicator(close,window,smooth1,smooth2).stochrsi_d()

def StochasticOscillator(close, high, low, window, smooth_window):
    return momentum.StochasticOscillator(close,high,low,window,smooth_window).stoch(), momentum.StochasticOscillator(close,high,low,window,smooth_window).stoch_signal()

def AwesomeOscillatorIndicator(high,low,window1,window2,fillna):
    return momentum.AwesomeOscillatorIndicator(high,low,window1,window2,fillna).awesome_oscillator()

def KAMAIndicator(close,window,pow1,pow2):
    return momentum.KAMAIndicator(close,window,pow1,pow2).kama()

def PercentagePriceOscillator(close,window_slow,window_fast,window_sign):
    return momentum.PercentagePriceOscillator(close,window_slow,window_fast,window_sign).ppo(), momentum.PercentagePriceOscillator(close,window_slow,window_fast,window_sign).ppo_signal(), momentum.PercentagePriceOscillator(close,window_slow,window_fast,window_sign).ppo_hist()

def PercentageVolumeOscillator(volume,window_slow,window_fast,window_sign):
    return momentum.PercentageVolumeOscillator(volume,window_slow,window_fast,window_sign).pvo(), momentum.PercentageVolumeOscillator(volume,window_slow,window_fast,window_sign).pvo_signal(), momentum.PercentageVolumeOscillator(volume,window_slow,window_fast,window_sign).pvo_hist()

def TSIIndicator(close, window_slow, window_fast):
    return momentum.TSIIndicator(close,window_slow,window_fast).tsi()

def UltimateOscillator(high,low,close,window1,window2,window3,weight1,weight2,weight3):
    return momentum.UltimateOscillator(high,low,close,window1,window2,window3,weight1,weight2,weight3).ultimate_oscillator()

def RIndicator(high,low,close,lbp):
    return momentum.WilliamsRIndicator(high,low,close,lbp).williams_r()


'''
Volume Indicators
'''
def AccDistIndexIndicator(high,low,close,vol):
    return volume.AccDistIndexIndicator(high,low,close,vol).acc_dist_index()

def ChaikinMoneyFlowIndicator(high,low,close,vol,window):
    return volume.ChaikinMoneyFlowIndicator(high,low,close,vol,window).chaikin_money_flow()

def EaseOfMovementIndicator(high,low,vol,window):
    return volume.EaseOfMovementIndicator(high,low,vol,window).ease_of_movement(), volume.EaseOfMovementIndicator(high,low,vol,window).sma_ease_of_movement()

def ForceIndexIndicator(close,vol,window):
    return volume.ForceIndexIndicator(close,vol,window).force_index()

def MFIIndicator(high,low,close,vol,window):
    return volume.MFIIndicator(high,low,close,vol,window).money_flow_index()

def NegativeVolumeIndexIndicator(close, vol):
    return volume.NegativeVolumeIndexIndicator(close,vol).negative_volume_index()

def OnBalanceVolumeIndicator(close,vol):
    return volume.OnBalanceVolumeIndicator(close,vol).on_balance_volume()

def VolumePriceTrendIndicator(close,vol):
    return volume.VolumePriceTrendIndicator(close,vol).volume_price_trend()

def VolumeWeightedAveragePrice(high,low,close,vol,window=14):
    return volume.VolumeWeightedAveragePrice(high,low,close,vol,window).volume_weighted_average_price()


'''
Volatility Indicators
'''
def AverageTrueRange(high,low,close,window): #Also an indicator
    return volatility.AverageTrueRange(high,low,close,window).average_true_range()

def BollingerBands(close,window,window_dev):
    x = volatility.BollingerBands(close,window,window_dev, fillna=True)
    return x.bollinger_hband(), x.bollinger_hband_indicator(), x.bollinger_lband(), x.bollinger_lband_indicator(), x.bollinger_pband(), x.bollinger_wband()

def DonchianChannel(high,low,close,window):
    x = volatility.DonchianChannel(high,low,close,window)
    return x.donchian_channel_hband(),x.donchian_channel_lband(),x.donchian_channel_mband(), x.donchian_channel_pband(),x.donchian_channel_wband()

#if original_version true, window_atr not valid as it uses SMA
# if its false, then window_atr is valid as it uses EMA
def KeltnerChannel(high,low,close,window,window_atr,original_version):
    x = volatility.KeltnerChannel(high,low,close,window,window_atr,original_version=original_version)
    return x.keltner_channel_hband(), x.keltner_channel_hband_indicator(),x.keltner_channel_lband(),x.keltner_channel_lband_indicator(), x.keltner_channel_mband(),x.keltner_channel_pband(),x.keltner_channel_wband()

def UlcerIndex(close,window):
    return volatility.UlcerIndex(close,window).ulcer_index()

'''
Trend Indicators
'''
#The Average Expanding Price range values, attempts to measure
#which highlights the trend in positive and negative directions 
#using Positive and Negative Directional Movement Indicators, 
#usually used for long term analysis and this indicator is prone to 
#producing false trading signals
def ADXIndicator(high,low,close,window):
    x = trend.ADXIndicator(high,low,close,window)
    return x.adx(), x.adx_neg(), x.adx_pos()
#
def AroonIndicator(close,window):
    x = trend.AroonIndicator(close,window)
    return x.aroon_down(), x.aroon_indicator(), x.aroon_up()

def CCIIndicator(high,low,close,window = 20,constant = 0.015,fillna = False):
    x = trend.CCIIndicator(high,low,close,window,constant,fillna)
    return x.cci()

def DPOIndicator(close, window = 20, fillna = False):
    x = trend.DPOIndicator(close,window,fillna)
    return x.dpo()

def EMAIndicator(close,window=20,fillna = False): 
    x = trend.EMAIndicator(close,window,fillna)
    return x.ema_indicator()

def IchimokuIndicator(high, low, window1 = 9, window2 = 26, window3 = 52, visual = False, fillna = False):
    x = trend.IchimokuIndicator(high,low,window1,window2,window3,visual,fillna)
    return x.ichimoku_a(), x.ichimoku_b(), x.ichimoku_base_line(), x.ichimoku_conversion_line()

def KSIIndicator(close,roc1=10, roc2=15, roc3=20, roc4 = 30, window1 = 10, window2 = 10, window3= 10, window4 = 15, nsig = 9, fillna = False):
    x = trend.KSTIndicator(close,roc1,roc2,roc3,roc4,window1,window2,window3,window4,nsig,fillna)
    return x.kst(), x.kst_diff(), x.kst_sig()

def MACD(close, window_slow = 26, window_fast = 12, window_sign = 9, fillna = False):
    x = trend.MACD(close,window_slow,window_fast,window_sign,fillna)
    return x.macd(), x.macd_diff(), x.macd_signal()

def MassIndex(high,low,window_fast=9,window_slow=25,fillna=False):
    x = trend.MassIndex(high,low,window_fast,window_slow,fillna)
    return x.mass_index()

def PSARIndicator(high,low,close,step = 0.02, max_step = 0.2, fillna = False):
    x = trend.PSARIndicator(high,low,close,step,max_step,fillna)
    return x.psar(), x.psar_down, x.psar_down_indicator(), x.psar_up, x.psar_up_indicator()

def SMAIndicator(close,window = 20, fillna = False):
    return trend.SMAIndicator(close,window,fillna).sma_indicator()

def STCIndicator(close,window_slow = 50, window_fast = 23, cycle = 10, smooth1 = 3,
                smooth2 = 3, fillna = False):
    
    x = trend.STCIndicator(close,window_fast,window_slow,cycle,smooth1,smooth2,fillna)
    return x.stc()

def TRIXIndicator(close, window = 15, fillna = False):
    return trend.TRIXIndicator(close,window,fillna).trix()

def VortexIndicator(high,low,close,window = 14, fillna = False):
    x = trend.VortexIndicator(high,low,close,window,fillna)
    return x.vortex_indicator_diff(), x.vortex_indicator_neg(), x.vortex_indicator_pos()

def WMAIndicator(close,window = 9, fillna = False):
    return trend.WMAIndicator(close,window,fillna).wma()
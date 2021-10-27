from datetime import date
from time import sleep
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, TIME_IN_FORCE_GTC
from matplotlib import colors
import pandas
import pandas as pd
from pandas.core.window import rolling
pd.options.mode.chained_assignment=None
from multiprocessing import Process, freeze_support, set_start_method
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import sqlalchemy
import numpy as np
#import pprint
from binance.client import Client
from binance import BinanceSocketManager
from shapely.geometry import LineString

#plt.ion()
class MyMoney:
    wallet = 10000
    buy_price = 0
    log_buy = 0
    Win_Count = 0
    Loss_Count =0
    profit = 0
    stoploss = 0
interval = '1000'   #Số lượng trích mẫu 
time_interval = '4' # Thời gian mỗi lần lấy trích mẫu (minute)
his= int(interval)*int(time_interval)  + int(time_interval)*2
api_key = "A6S9QsiqeOahLIOsfdxziQLIT2TzJ4ADr08NbPWumcPPezRWCL7KaicEuD624rNH"
api_scret = "BDKQqtvFzSEt9ADu9oPCKuM1fVql3RvSr51qOODvCQV8sFDEu3n6D5FSvY5pgxe5"
client = Client(api_key,api_scret)
def getminutedata(symbol,inter,lookback):   
    frame = pd.DataFrame(client.get_historical_klines(symbol,inter,lookback))
    frame =frame.iloc[:,:5]
    frame.columns = ['Time','Open','High','Low','Close']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit ='ms')
    frame = frame.astype(float)
    return frame
def getcurrentprice(sym):
    #btc_price = client.get_symbol_ticker(symbol="BTCUSDT")
    frame = pd.DataFrame(client.get_symbol_ticker(symbol = "BTCUSDT" ))
    # frame =frame.iloc[:,:5]
    # frame.columns = ['Time','Open','High','Low','Close']
    # frame = frame.set_index('Time')
    # frame.index = pd.to_datetime(frame.index, unit ='ms')
    frame = frame.astype(float)
    return frame
#data = client.get_symbol_ticker(symbol = "CELOUSDT" )   # Lấy Giá Hiện Tại ------------------
#print(client.get_symbol_ticker(symbol = "CELOUSDT" ))
def Buy(sym,usd):
    curr_price = client.get_symbol_ticker(symbol = "CELOUSDT" )
    order = client.create_order(symbol = sym,side = 'BUY',type = 'MARKET',quantity = float(round(usd/float(curr_price['price']))))
    return order

def Sell(sym,coin_name):
    #curr_price = client.get_symbol_ticker(symbol = sym )
    balance= client.get_asset_balance(asset=coin_name)
    order = client.create_order(symbol = sym,side = 'SELL',type = 'MARKET',quantity = float(balance['free']))
    return order

while False:
    try :
        data = client.get_symbol_ticker(symbol = "BTCUSDTl" )
        print(data['price'])       
    except:
        print('check again')
        sleep(0.1)

Account = client.get_account()   # Lấy dữ liệu ng dùng
balance= client.get_asset_balance(asset='CELO') # Lấy số d coin mong muốn
print(balance)
#print(Buy('CELOUSDT',42.6)) 
#print(Sell('CELOUSDT','CELO'))    



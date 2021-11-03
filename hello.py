from datetime import date
from time import sleep
from matplotlib import colors, ticker, transforms
from matplotlib import lines
import pandas
import pandas as pd
pd.set_option('display.max_rows', None)
from pandas.core import window
from pandas.core.window import rolling
pd.options.mode.chained_assignment=None
from multiprocessing import Process, freeze_support, set_start_method
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import sqlalchemy
import numpy as np
from binance.client import Client
from binance import BinanceSocketManager
from shapely.geometry import LineString
from scipy.signal import savgol_filter
import threading
import logging

#plt.ion()
class MyMoney:
    wallet = 10000
    buy_price = 0
    log_buy = 0
    Win_Count = 0
    Loss_Count =0
    profit = 0
    stoploss = 0
    start_cap = 1000
    d_test = []
coinname = 'SHIB'
symbol = coinname +'USDT'
interval = '100'   #Số lượng trích mẫu 
time_interval = '4' # Thời gian mỗi lần lấy trích mẫu (minute)
his= int(interval)*int(time_interval)  + int(time_interval)*2
api_key = "A6S9QsiqeOahLIOsfdxziQLIT2TzJ4ADr08NbPWumcPPezRWCL7KaicEuD624rNH"
api_scret = "BDKQqtvFzSEt9ADu9oPCKuM1fVql3RvSr51qOODvCQV8sFDEu3n6D5FSvY5pgxe5"
client = Client(api_key,api_scret)
def buy(curr_wallet,Monney):   
    curr = curr_wallet - Monney
    return curr
def sell(buy_price, curr_price, curr_wallet,Money):
    a= curr_price*Money/buy_price + curr_wallet
    return a
def getminutedata(sym,inter,lookback1,lookback2):   
    #frame = pd.DataFrame(client.get_historical_klines(symbol,inter,lookback1,lookback2))
    frame = pd.DataFrame(client.get_historical_klines(sym,inter,lookback1))
    frame =frame.iloc[:,:8]
    frame.columns = ['Time','Open','High','Low','Close','Volume','Close_Time','Quote_Volumn']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit ='ms')
    frame = frame.astype(float)
    return frame

def getcurrentprice(sym):
    frame = pd.DataFrame(client.get_symbol_ticker(symbol = sym))
    
    frame = frame.astype(float)
    return frame

def create_symbols_list(filter='USDT'):
    rows = []
    info = pd.DataFrame(client.get_all_isolated_margin_symbols())
    info=info.loc[info['quote'] == 'USDT']
    pairs_data = info.iloc[:]['symbol']
    full_data = pairs_data[pairs_data.str.contains(filter)]
    info['symbol'].apply(lambda x: x==full_data)
    return full_data

def get_Product(filter='USDT'):
    rows = []
    info = pd.DataFrame(client.get_products()['data'])
    info=info.loc[info['q'] == 'USDT']
    rows = pd.DataFrame(rows,columns=['s','name','cs','c','cap'])
    rows['s']=info.s
    rows['name']=info.b
    rows['cs']=info.cs
    rows['c']=info.c
    rows['cap']=info.c * info.cs
    rows = rows.loc[(rows['name'] != 'USDC') & (rows['name'] != 'USDS') & (rows['name'] != 'BUSD') ]
    rows = rows.nlargest(20, ['cap']).reset_index() 
    return rows
print (get_Product())
list_coin = list(create_symbols_list('USDT'))


# removes 11, 18, 23
unwanted = ['USDCUSDT', 'USDSUSDT', 'USDSBUSDT', 'BUSDUSDT','EURUSDT', 'TUSDUSDT','LENDUSDT','NPXSUSDT', 'XZCUSDT']
for i in range (0,(int(len(list_coin))-1) ):
    
    c_name = list_coin[i]
    if (c_name[len(c_name)-4] + c_name[len(c_name)-3] + c_name[len(c_name)-2] + c_name[len(c_name)-1])!= 'USDT' :
        unwanted.append(c_name)

for i in range (0,(int(len(unwanted)))):  
    if unwanted[i] in list_coin :
        list_coin.remove(unwanted[i])
    
#print ('---------------',list_coin,len(list_coin),'----')

def filter(df,list):
    Rows=[]
    for i in range(0,len(list)):          
        df2  = getminutedata(list[i],Client.KLINE_INTERVAL_1DAY,'1 days ago UTC',"28 OCT, 2021")
        Rows.append([df2.iloc[0]['Volume'],list[i],df2.iloc[0]['Volume'] * df2.iloc[0]['Close'] ])
    df = pd.DataFrame(Rows,columns=['Volume','Symbol','Cap'])
    df = df.nlargest(10, ['Cap']).reset_index() 
    print(df)
    return df    
#filter(MyMoney.d_test,list_coin)
print('---------------------')


def Real_Buy(sym,usd):      #usd: Số tiền usd bỏ vào mua
    curr_price = client.get_symbol_ticker(symbol = "CELOUSDT" )
    order = client.create_order(symbol = sym,side = 'BUY',type = 'MARKET',quantity = float(round(usd/float(curr_price['price']))))
    return order

def Real_Sell(sym,coin_name):
    #curr_price = client.get_symbol_ticker(symbol = sym )
    balance= client.get_asset_balance(asset=coin_name)
    order = client.create_order(symbol = sym,side = 'SELL',type = 'MARKET',quantity = float(balance['free']))
    return order


def acount (coin_name):
    Account = client.get_account()   # Lấy dữ liệu ng dùng
    balance= client.get_asset_balance(asset=coin_name) # Lấy số dư coin mong muốn
    return Account,balance


    #---------------------------------------------------------------------------------------Crawl Data ------------------------------------------------
def Crawl_Data(sym):
        
    #data = getminutedata('BTCUSDT',time_interval+'m',str(his)+' min ago UTC')
    data = getminutedata(sym,Client.KLINE_INTERVAL_4HOUR,'2004 hours ago UTC',"28 OCT, 2021")
    #data = getminutedata(symbol,Client.KLINE_INTERVAL_4HOUR,"1 JAN, 2021", "28 OCT, 2021")
    
    #----------------------------------------------------------------------- ADX -----------------------------------------------------------------------

    def ADX(df,index_column):       
        df['+DM']=np.NaN #[4]
        df['-DM']=np.NaN  #[4+1]
        df['TR']=np.NaN  #[4+2]
        df['TR14']=np.NaN  #[4+3]
        df['+DM14']=np.NaN   #[4+4]
        df['-DM14']=np.NaN  #[4+5]
        df['+DI']=np.NaN  #[4+6]
        df['-DI']=np.NaN   #[4+7]
        df['DX']=np.NaN  #[4+8]
        df['ADX'] = np.NaN #[4+9]
        cDmp = index_column   # Column number for idoc[,]
        cDmn = cDmp +1
        cTr = cDmp +2
        cTr14 = cDmp +3
        cDm14p = cDmp +4
        cDm14n = cDmp +5
        cDip = cDmp +6
        cDin = cDmp +7
        cDx = cDmp +8
        cAdx = cDmp +9                
        def calc_val(df, column,index):
                prev_val = df.iloc[index-1][column]
                curr_val = df.iloc[index][column]
                return(curr_val, prev_val)

        def calc_dm(df, index):
                curr_high, prev_high = calc_val(df, 'High',index)
                curr_low, prev_low = calc_val(df, 'Low',index)

                dm_pos = curr_high - prev_high
                dm_neg = prev_low - curr_low
                
                if dm_pos > dm_neg:
                    if dm_pos < 0:
                        dm_pos = 0.00
                    dm_neg = 0.00
                    return(dm_pos, dm_neg)

                elif dm_pos < dm_neg:
                    if dm_neg < 0:
                        dm_neg = 0.00
                    dm_pos = 0.00
                    return(dm_pos, dm_neg)
                
                else:
                    if dm_pos < 0:
                        dm_pos = 0.00
                    dm_neg = 0.00
                    return(dm_pos, dm_neg)

        def calc_tr(df, index):
                curr_high, prev_high = calc_val(df, 'High',index)
                curr_low, prev_low = calc_val(df, 'Low',index)
                curr_close, prev_close = calc_val(df, 'Close',index)
                ranges = [curr_high - curr_low, abs(curr_high - prev_close), abs(curr_low - prev_close)]
                TR = max(ranges)
                return(TR)

        def calc_first_14(df, index, column):
                result = 0
                for a in range(index-13, index+1):
                    result += df.iloc[a][column]
                return(result)

        def calc_subsequent_14(df, index, column):
                return(df.iloc[index-1][column+'14'] - (df.iloc[index-1][column+'14']/14) + df.iloc[index][column])

        def calc_first_adx(df, index):
                result = 0
                for a in range(index-13, index+1):
                    result += int(df.iloc[a]['DX'])
                return(result/14)

        def calc_adx(df, index):
                return(round(((df.iloc[index-1]['ADX']*13) + df.iloc[index]['DX'])/14, 2))        
        for i in range(1, int(interval)):
            dm_pos, dm_neg = calc_dm(df, i)
            TR = calc_tr(df, i)   
            df.iloc[i,cDmp]=dm_pos
            df.iloc[i,cDmn] = dm_neg
            df.iloc[i,cTr] = TR
            
            if df[df.columns[cTr]].count() == 14:
                df.iloc[i, cTr14] = calc_first_14(df, i, 'TR')
                df.iloc[i, cDm14p] = calc_first_14(df, i, '+DM')
                df.iloc[i, cDm14n] = calc_first_14(df, i, '-DM')
            
            elif df[df.columns[cTr]].count() >= 14:   
                df.iloc[i, cTr14] = round(calc_subsequent_14(df, i, 'TR'),2)
                df.iloc[i, cDm14p] = round(calc_subsequent_14(df, i, '+DM'), 2)
                df.iloc[i, cDm14n] = round(calc_subsequent_14(df, i, '-DM'), 2)

            if df[df.columns[cTr14]].count() >= 1:
                df.iloc[i, cDip] = round((df.iloc[i, cDm14p] / df.iloc[i, cTr14])*100, 2)
                df.iloc[i, cDin] = round((df.iloc[i, cDm14n] / df.iloc[i, cTr14])*100, 2)
                df.iloc[i, cDx] = round((abs(df.iloc[i, cDip] - df.iloc[i, cDin])/abs(df.iloc[i,cDip] + df.iloc[i,cDin]) )*100 , 2)
            if df[df.columns[cDx]].count()==14:
                    df.iloc[i, cAdx] = calc_first_adx(df, i)
                
            elif df[df.columns[cDx]].count()>=14:
                    df.iloc[i, cAdx] = calc_adx(df, i)
           
        return df       
    #ADX(data,5)
    
    #    
    #---------------------------------------------------------------------SMA & EMA --------------------------------------------------------------------------------------
    def MA(df,number):
        df['MA'+str(number)] = df['Close'].rolling(number).mean()
        return df
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA200'] = data['Close'].rolling(200).mean()
    
    def EMA(df,number):
        data['EMA'+str(number)] = data['Close'].ewm(span=number,adjust=False).mean()
    #EMA(data,20)
    #data['EMA100'] = data['Close'].ewm(span=100,adjust=False).mean()
    #print(data.tail(50))
    #---------------------------------------------------------------------- MACD --------------------------------------------------------------------------------------
    def MACD(df):
        exp1 = df['Close'].ewm(span= 12 , adjust= False).mean()
        exp2 = df['Close'].ewm(span= 26 , adjust= False).mean()
        df['MACD']=exp1 -exp2
        df['Signal']  = df['MACD'].ewm(span= 9 , adjust= False).mean()
        return df
    #MACD(data)

    #---------------------------------------------------------------------- Stochastic Oscillator ------------------------------------------------------------------------
    def Stochartic(df):
        high14 = df['High'].rolling(14).max()
        low14 = df['Low'].rolling(14).min()
        df['%K'] = (df['Close']-low14)*100/(high14-low14)
        df['%D'] = df['%K'].rolling(3).mean()
        return df
    #Stochartic(data)
    #---------------------------------------------------------------------- RSI ------------------------------------------------------------------------------------------
    def RSI(df):
        delta =  df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up= up.ewm(com=13, adjust=False).mean()
        ema_down=down.ewm(com=13,adjust=False).mean()
        rs = ema_up/ema_down
        df['RSI'] = 100 -(100/(1+rs))
        return df
    #RSI(data)

    #---------------------------------------------------------------------- CCI ------------------------------------------------------------------------------------------
    def CCI(df, ndays): 
        TP = (df['High'] + df['Low'] + df['Close']) / 3 
        SMA = TP.rolling(ndays).mean()
        MAD = TP.rolling(ndays).apply(lambda x: pd.Series(x).mad())
        df['CCI'] = (TP - SMA) / (0.015 * MAD) 
        return df
    CCI(data,14)

    #---------------------------------------------------------------------- Ichimoku ------------------------------------------------------------------------------------------
    def ichimoku(df):
        High9 = df.High.rolling(9).max()
        Low9 = df.High.rolling(9).min()
        High26 = df.High.rolling(26).max()
        Low26 = df.High.rolling(26).min()
        High52 = df.High.rolling(52).max()
        Low52 = df.High.rolling(52).min()

        df['Tenkan_sen']=(High9+Low9)/2
        df['kijun_sen']=(High26+Low26)/2
        df['Senkou_A']=((df['Tenkan_sen']+ df['kijun_sen'])/2).shift(26)
        df['Senkou_B']=((High52+Low52)/2).shift(26)
        df['Chikou']=df['Close'].shift(-26)
        return df
    #ichimoku(data)

#---------------------------------------------------------------------- Bollinger Bands ------------------------------------------------------------------------------------------
    def bollinger_bands(df):
        SMA = df['Close'].rolling(window= 20).mean()
        stddev = df['Close'].rolling(window= 20).std()
        df['Upper'] = SMA + 2*stddev
        df['Low'] = SMA - 2*stddev
        return df
    #bollinger_bands(data)

#-----------------------------------------------------------------------Fibonachi -------------- Chưa Viết Xong---------------------------------------------
    # The fibonachi ratios are 0.236,0.382 and 0.618
    #Example 1,1,2,3,5,8,13,21,34,55,89,144 => 89/144 = 0.618
    # To get 0.382 take any number in the Fibonachi squence and divide it by the next number in the sequence
    # Example 34/89 = 0.382

    # First get max and min close price for time period
    def fibonachi(df):
        max_price = df['Close'].max()
        min_price = df['Close'].min()
        diffirence = max_price - min_price
        first_level = max_price- diffirence*0.236
        Second_level = max_price- diffirence*0.382
        third_level = max_price- diffirence*0.5
        fourth_level = max_price- diffirence*0.618


    def fibonachi(df):
        sma350 = MA(df,350)['MA350']
        sma111 = MA(df,111)['MA111']

#------------------------------------------------------------------------DPO Detrended Price Oscillator-----------------------------------------------------------
    
    def detrended_price_oscillator(df,period):
        
        # Calculating the Simple Moving Average 
        MA(data,period)        
        df['DPO'] = df.Close - df.MA20.shift(periods =int(period/2 +1))       
        return df
#-----------------------------------------------------------------------Super Trend -----------------------------------------------------------
    
    def tr(df):
        df['pre_close'] = df['Close'].shift(1)
        df['high_low']=abs(df['High']-df['Low'])
        df['high_pc']=abs(df['High']-df['pre_close'])
        df['low_pc']=abs(df['Low']-df['pre_close'])
        tr = df[['high_low','high_pc','low_pc']].max(axis =1)
        return tr
    
    def supertrend(df, period, atr_multiplier):
        tr1 =tr(df)
        atr = tr1.rolling(period).mean()
        hl2 = (df['High'] + df['Low']) / 2
        #df['atr'] = atr(df, period)
        upperband = hl2 + (atr_multiplier * atr)
        lowerband = hl2 - (atr_multiplier * atr)
        df['in_uptrend'+ str(period)] = True

        for current in range(1, len(df.index)):
            previous = current - 1

            if df['Close'][current] > upperband[previous]:
                df['in_uptrend'+ str(period)][current] = True
            elif df['Close'][current] < lowerband[previous]:
                df['in_uptrend'+ str(period)][current] = False
            else:
                df['in_uptrend'+ str(period)][current] = df['in_uptrend'+ str(period)][previous]

                if df['in_uptrend'+ str(period)][current] and lowerband[current] < lowerband[previous]:
                    lowerband[current] = lowerband[previous]

                if not df['in_uptrend'+ str(period)][current] and upperband[current] > upperband[previous]:
                    upperband[current] = upperband[previous]
            
        return df
    supertrend(data,10,3) 
    #supertrend(data,20,5)
    supertrend(data,20,6)
    #SuperTrend(data,10,3)
    data['Index']= range(len(data))

    data = data.dropna()   # Lọc những dữ liệu NaN
    #data['price change'] = data['Close'].pct_change()
    #data = data.dropna()   # Lọc những dữ liệu NaN
    #print(data.tail(50))
    #print(data.head(50))
    return data

#-----------------------------------------------------------------Candlestick Chart --------------------------------Chưa viết xong ---------------------------------------------
def candle_chart (df):
    plt.style.use('ggplot')
    fig,ax = plt.subplots()
  #  candlestick_ohlc(ax, df.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    print(df)
    
#-----------------------------------------------------------------Graph Trend ------------------------------------------------------------------------------------
def EMA10_graph(df):
    fig, ax = plt.subplots()
    df[['EMA10']].plot(ax=ax)
    plt.show()
def ADX_graph(df):
    fig, ax = plt.subplots()
    df[['+DI','-DI','ADX']].plot(ax=ax)
    plt.show()

def suport_resistance(df):
        pivot_high_1= df['High'][-50:-1].max()
        pivot_high_2= df['High'][-100:-51].max()

        pivot_low_1= df['Low'][-50:-1].min()
        pivot_low_2= df['Low'][-100:-51].min()

        A= [df['High'][-50:-1].idxmax(),pivot_high_1]
        B= [df['High'][-100:-51].idxmax(),pivot_high_2]

        A1= [df['Low'][-50:-1].idxmax(),pivot_low_1]
        B1= [df['Low'][-100:-51].idxmax(),pivot_low_2]

        x1_high_value=[A[0],B[0]]
        y1_high_value=[A[1],B[1]]

        x1_low_value=[A1[0],B1[0]]
        y1_low_value=[A1[1],B1[1]]

        plt.rcParams.update({ 'font.size':10 })
        fig,ax1 = plt.subplots(figsize = (14,7))

        ax1.set_ylabel('aaa')
        ax1.set_xlabel('Date')
        ax1.set_title('bbb')
        ax1.plot('Close',data = df ,linewidth=0.5, color = 'blue')
        ax1.plot(x1_high_value,y1_high_value,color='g',linestyle='--',linewidth=0.5 )
        ax1.plot(x1_low_value,y1_low_value,color='r',linestyle='--',linewidth=0.5)

        ax1.axhline(y=pivot_high_1,color = 'g', linewidth = 6, alpha = 0.2)
        ax1.axhline(y=pivot_low_1,color = 'r', linewidth = 6, alpha = 0.2)

        trans = transforms.blended_transform_factory(ax1.get_yticklabels()[0].get_transform(),ax1.transData)

        ax1.text(0,pivot_high_1,"{:.2f}".format(pivot_high_1),color='g',transform=trans, ha='right',va='center')
        ax1.text(0,pivot_low_1,"{:.2f}".format(pivot_low_1),color='g',transform=trans, ha='right',va='center')
        ax1.legend()
        ax1.grid()
        plt.show()

def suport_resistance2(df):
    fee = 0.0005
    sr_sell = 0.7
    sr_buy = 0.3
    #coding technical analysis signals\n,
    df['returns'] = df['Close'].pct_change()
    df['scaled price'] = df['Close']/10**np.floor(np.log10(df['Close']))
    df['S&R'] = df['scaled price']%1
    #simulating trading strategies\n,
    df['signal'] = 1*(df['S&R'] < sr_buy) - 1*(df['S&R'] > sr_sell)
    BnH_return = np.array(df['returns'][1:])
    SR_return = np.array(df['returns'][1:])*np.array(df['signal'][:-1]) - fee*abs(np.array(df['signal'][1:])-np.array(df['signal'][:-1]))
    BnH = np.prod(1+BnH_return)**(252/len(BnH_return)) - 1
    SR = np.prod(1+SR_return)**(252/len(SR_return)) - 1
    BnH_risk = np.std(BnH_return)*(252)**(1/2)
    SR_risk = np.std(SR_return)*(252)**(1/2)
    #visualising the results\n,
    print('buy-and-hold strategy return and risk: '+str(round(BnH*100,2))+'% and '+str(round(BnH_risk*100,2))+'%')
    print('support and resistance strategy return and risk: '+str(round(SR*100,2))+'% and '+str(round(SR_risk*100,2))+'%')
    #plt.plot('Close',data = df ,linewidth=0.5, color = 'blue')
    plt.plot(np.append(1,np.cumprod(1+BnH_return)))
    plt.plot(np.append(1,np.cumprod(1+SR_return)))

x_values = []
y_values = []

plt.style.use('fivethirtyeight')
fig = plt.figure()

def pt_AB(x0,y0,x1,y1):
    a= (y1-y0)/(x1-x0)
    b= y1-a*x1
    return a,b
def pt_intersection(a0,b0,a1,b1):
    x= (b1-b0)/(a0-a1)
    y = a0*x + b0
    return x,y

#------------------------------------------------------------Candle-------------------------0:N/A     1: Up   -1: Down-------------

def harami_up(index,df):    #mang thai
    
    if (df.iloc[index-1]['Close'] < df.iloc[index-1]['Open'] and df.iloc[index]['Close'] > df.iloc[index]['Open'] and
        df.iloc[index]['Open'] >= df.iloc[index]['Low'] > df.iloc[index-1]['Close'] ):
            return 1
    elif (df.iloc[index-1]['Close'] > df.iloc[index-1]['High'] and df.iloc[index]['Close'] < df.iloc[index]['Open'] and
            df.iloc[index-1]['Close'] > df.iloc[index]['High'] > df.iloc[index-1]['Close'] ):
            return -1
    else : return 0
          
def windows_gaps(index,df):    # khoang trong
    if  df.iloc[index]['Low'] > df.iloc[index-1]['High']:
        return 1
    elif df.iloc[index-1]['Low'] > df.iloc[index]['High']:
        return -1
    else : return 0

def star(index,df):    # evening star & morning star
    if  (df.iloc[index-2]['Close'] > df.iloc[index-2]['Open'] and 
        df.iloc[index-1]['Close'] < df.iloc[index-1]['Open'] and df.iloc[index-1]['Close'] < df.iloc[index-2]['Close'] and df.iloc[index-1]['Close'] > 0.5*abs(df.iloc[index-2]['Open']-df.iloc[index-2]['Close']) and
        df.iloc[index]['Close'] < df.iloc[index]['Open'] and df.iloc[index]['Close'] < 0.5 * abs(df.iloc[index-2]['Close'] - df.iloc[index-2]['Open']) ):        
        return -1
    elif  (df.iloc[index-2]['Close'] < df.iloc[index-2]['Open'] and 
        df.iloc[index-1]['Close'] > df.iloc[index-1]['Open'] and df.iloc[index-2]['Close'] < df.iloc[index-1]['Close'] and df.iloc[index-1]['Close'] < 0.5*abs(df.iloc[index-2]['Open']-df.iloc[index-2]['Close']) and
        df.iloc[index]['Close'] > df.iloc[index]['Open'] and df.iloc[index]['Close'] > 0.5 * abs(df.iloc[index-2]['Close'] - df.iloc[index-2]['Open']) ):
        return 1       
    else : return 0

def engulfing(index,df):        # nen nhan chim
    if  (df.iloc[index-1]['Close'] < df.iloc[index-1]['Open'] and 
        df.iloc[index]['Close'] > df.iloc[index]['Open'] and df.iloc[index-1]['Close'] > df.iloc[index]['Open'] ) :  
        return 1
    if  (df.iloc[index-1]['Close'] > df.iloc[index-1]['Open'] and 
        df.iloc[index]['Close'] < df.iloc[index]['Open'] and df.iloc[index-1]['Close'] < df.iloc[index]['Open'] ) : 
        return -1       
    else : return 0

def three_methods(index,df) :              # mo hinh 3 nen tang giam
    if  (df.iloc[index-4]['Close'] > df.iloc[index-4]['Open'] and 
        df.iloc[index-3]['Close'] < df.iloc[index-3]['Open'] and df.iloc[index-4]['Close'] > df.iloc[index-3]['Close'] and
        df.iloc[index-2]['Close'] < df.iloc[index-2]['Open'] and df.iloc[index-3]['Close'] > df.iloc[index-2]['Close'] and
        df.iloc[index-1]['Close'] < df.iloc[index-1]['Open'] and df.iloc[index-2]['Close'] > df.iloc[index-1]['Close'] and
        df.iloc[index]['Close'] > df.iloc[index]['Open'] and df.iloc[index]['Close'] > df.iloc[index-4]['Close'] and
        df.iloc[index-1]['Close'] > df.iloc[index-4]['Open'] ) :       
        return 1
    elif  (df.iloc[index-4]['Close'] < df.iloc[index-4]['Open'] and 
        df.iloc[index-3]['Close'] > df.iloc[index-3]['Open'] and df.iloc[index-4]['Close'] < df.iloc[index-3]['Close'] and
        df.iloc[index-2]['Close'] > df.iloc[index-2]['Open'] and df.iloc[index-3]['Close'] < df.iloc[index-2]['Close'] and
        df.iloc[index-1]['Close'] > df.iloc[index-1]['Open'] and df.iloc[index-2]['Close'] < df.iloc[index-1]['Close'] and
        df.iloc[index]['Close'] < df.iloc[index]['Open'] and df.iloc[index]['Close'] < df.iloc[index-4]['Close'] and
        df.iloc[index-1]['Close'] < df.iloc[index-4]['Open'] ) :       
        return -1       
    else : return 0

def doji(index,df):                      # Chưa viết xong
    if (0.997 <df.Close[index] / df.Open[index] < 1.003 and (df.Close[index] - df.Low[index])/(df.High[index]-df.Open[index]) > 4 and 
        df.Close[index-1]< df.Open[index-1] ) :
        return 1
        
def check_candle(index,df):
    a = harami_up(index,df) 
    b = windows_gaps(index,df)
    c = star(index,df)
    d = engulfing(index,df)
    e = three_methods(index,df)
    f = 0
    f = a + b + c + d 
    return f

def main(i):       
    avg_profit =[]
    Listcoin=list(get_Product().s)
    print(Listcoin)
    #print(update_data.tail(10))
    # plt.cla() 
    # plt.plot(update_data.index,update_data.EMA20,linewidth=1.0,color='green')
    # plt.plot(update_data.index,update_data.Close,linewidth=1.0,color='green')
    #top10_coin = filter(MyMoney.d_test,list_coin)
    for a in range(0,len(Listcoin)-1) :
        print(Listcoin[a])
        update_data = Crawl_Data(Listcoin[a])
        MyMoney.wallet = 10000
        MyMoney.Win_Count = 0
        MyMoney.buy_price = 0
        MyMoney.log_buy =0
        MyMoney.Loss_Count = 0
        MyMoney.stoploss = 0
        MyMoney.profit = 0
        MyMoney.start_cap = 1000
        
        for i in range(20,len(update_data)):       
            # if update_data.iloc[i-1]['in_uptrend10'] != update_data.iloc[i]['in_uptrend10']:
            #     print(update_data.index[i])
            #     plt.plot(update_data.index[i], update_data.iloc[i]['Close'], 'ro')   # cham diem tren so do
                        
            if (update_data.iloc[i-2]['CCI']>0>update_data.iloc[i-1]['CCI'] or update_data.iloc[i]['Low'] < MyMoney.stoploss ) and  MyMoney.buy_price >0:
                if  update_data.iloc[i]['Low'] < MyMoney.stoploss :
                    MyMoney.wallet = sell(MyMoney.buy_price,MyMoney.stoploss,MyMoney.wallet,MyMoney.start_cap)
                elif  update_data.iloc[i-2]['CCI']>0>update_data.iloc[i-1]['CCI'] :
                    MyMoney.wallet = sell(MyMoney.buy_price,update_data.iloc[i]['Open'],MyMoney.wallet,MyMoney.start_cap)
                        
                if update_data.Open[i] >=  MyMoney.buy_price : MyMoney.Win_Count+= 1
                if update_data.Open[i] <  MyMoney.buy_price : MyMoney.Loss_Count+= 1
                #print(MyMoney.buy_price ,'-----', update_data.Open[i],'-----',MyMoney.log_buy,'-----', i, '--Win--',MyMoney.Win_Count,'--Loss--',MyMoney.Loss_Count,'-----',abs(update_data.Open[i]-MyMoney.buy_price),'------',MyMoney.wallet)
                MyMoney.buy_price, MyMoney.log_buy = 0,0     
            if  update_data.iloc[i-2]['CCI']<100 < update_data.iloc[i]['CCI'] and update_data.iloc[i-1]['in_uptrend10'] == update_data.iloc[i-1]['in_uptrend20'] == True and MyMoney.buy_price ==0 :
                    if MyMoney.wallet >= 10000 :
                        MyMoney.start_cap = MyMoney.wallet - 9000
                    else :
                        MyMoney.start_cap = 1000
                    MyMoney.wallet = buy(MyMoney.wallet,MyMoney.start_cap)
                    MyMoney.buy_price = update_data.iloc[i]['Open']
                    MyMoney.stoploss = 0.8*MyMoney.buy_price
                    MyMoney.log_buy = i
                    #print('---------------',update_data.index[i])                           
            
        if(MyMoney.buy_price > 0):
            MyMoney.wallet = MyMoney.wallet + MyMoney.start_cap
            MyMoney.buy_price = 0
        print(MyMoney.wallet,'-------',a,'--win: ',MyMoney.Win_Count,'--loss: ',MyMoney.Loss_Count,'---',update_data.Volume[len(update_data)-1])
        avg_profit.append(MyMoney.wallet)
        # plt.xlabel('Time')
        # plt.ylabel('Price')
        # plt.title('Name')
        # plt.gcf().autofmt_xdate()
        # plt.tight_layout()
    print (np.mean(avg_profit))
#ani = FuncAnimation(plt.gcf(),main,interval=10000)         
# plt.tight_layout()
# plt.show()

while True:
    main(1)
    #print (client.get_symbol_ticker(symbol = symbol )['price'])
    #print (coinname)
    print('--------------------------------------------------------------')
    #sleep(1)
    

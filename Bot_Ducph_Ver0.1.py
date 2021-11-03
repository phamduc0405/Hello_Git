from datetime import date
from time import sleep
import pandas as pd
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment=None
from pandas.core import window
from pandas.core.window import rolling
import numpy as np
from binance.client import Client
from binance import BinanceSocketManager
from shapely.geometry import LineString
from scipy.signal import savgol_filter

class MyMoney:
    wallet = 10000
    buy_price = 0
    log_buy = 0
    Win_Count = 0
    Loss_Count =0
    profit = 0
    stoploss = 0
    start_cap = 1000
api_key = "A6S9QsiqeOahLIOsfdxziQLIT2TzJ4ADr08NbPWumcPPezRWCL7KaicEuD624rNH"
api_scret = "BDKQqtvFzSEt9ADu9oPCKuM1fVql3RvSr51qOODvCQV8sFDEu3n6D5FSvY5pgxe5"
client = Client(api_key,api_scret)
interval = '100'   #Số lượng trích mẫu 
def buy(curr_wallet,Monney):   
    curr = curr_wallet - Monney
    return curr
def sell(buy_price, curr_price, curr_wallet,Money):
    a= curr_price*Money/buy_price + curr_wallet
    return a
def getminutedata(sym,inter,lookback1,lookback2):   
    frame = pd.DataFrame(client.get_historical_klines(sym,inter,lookback1,lookback2))
    #frame = pd.DataFrame(client.get_historical_klines(sym,inter,lookback1))
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
#------------------------------------------------------------------------------- Lấy USDT & Top 20 ------------------------------------------
def get_Product(filter='USDT'):
    rows = []
    #info1 = pd.DataFrame(client.get_historical_trades(symbol='BTCUSDT')) 
    #info1 = pd.DataFrame(client.aggregate_trade_iter(symbol='BTCUSDT', start_str='30 minutes ago UTC')) 
    info = pd.DataFrame(client.get_products()['data'])
    info=info.loc[info['q'] == filter]
    rows = pd.DataFrame(rows,columns=['s','name','cs','c','cap'])
    rows['s']=info.s
    rows['name']=info.b
    rows['cs']=info.cs
    rows['c']=info.c
    rows['cap']=info.c * info.cs
    rows = rows.loc[(rows['name'] != 'USDC') & (rows['name'] != 'USDS') & (rows['name'] != 'BUSD') ]
    rows = rows.nlargest(20, ['cap']).reset_index() 
    return rows

#---------------------------------------------------------------------------------------Crawl Data ------------------------------------------------
def Crawl_Data(sym):
        
    #data = getminutedata('BTCUSDT',time_interval+'m',str(his)+' min ago UTC')
    #data = getminutedata(sym,Client.KLINE_INTERVAL_4HOUR,'4004 hours ago UTC',"28 OCT, 2021")
    data = getminutedata(sym,Client.KLINE_INTERVAL_4HOUR,"1 JAN, 2020", "28 OCT, 2021")
    
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
    #---------------------------------------------------------------------SMA & EMA --------------------------------------------------------------------------------------
    def MA(df,number):
        df['MA'+str(number)] = df['Close'].rolling(number).mean()
        return df
    #MA(data,30)
    def EMA(df,number):
        data['EMA'+str(number)] = data['Close'].ewm(span=number,adjust=False).mean()
    #EMA(data,20)
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
    RSI(data)

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
    supertrend(data,20,6)
    data['Index']= range(len(data))
    data = data.dropna()   # Lọc những dữ liệu NaN
    return data
    
#--------------------------------------------------------------------------------------- Main Cycle ------------------------------------------------
def cap10_main():                 #--------------------- Lấy Dữ Top 10 Coin có Vốn Hóa Thị Trường Lớn
    avg_profit =[]
    Listcoin=list(get_Product().s)
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
            if (update_data.iloc[i-2]['CCI']>0>update_data.iloc[i-1]['CCI'] or update_data.iloc[i]['Low'] < MyMoney.stoploss or update_data.iloc[i]['Low'] < 0.85*update_data.iloc[i-1]['Close']  ) and  MyMoney.buy_price >0:
                if  update_data.iloc[i]['Low'] < MyMoney.stoploss :
                    MyMoney.wallet = sell(MyMoney.buy_price,MyMoney.stoploss,MyMoney.wallet,MyMoney.start_cap)
                elif  update_data.iloc[i-2]['CCI']>0>update_data.iloc[i-1]['CCI'] :
                    MyMoney.wallet = sell(MyMoney.buy_price,update_data.iloc[i]['Open'],MyMoney.wallet,MyMoney.start_cap)
                elif update_data.iloc[i]['Low'] < 0.85*update_data.iloc[i-1]['Close'] :
                    MyMoney.wallet = sell(MyMoney.buy_price,0.85*update_data.iloc[i-1]['Close'],MyMoney.wallet,MyMoney.start_cap)
                        
                if update_data.Open[i] >=  MyMoney.buy_price : MyMoney.Win_Count+= 1
                if update_data.Open[i] <  MyMoney.buy_price : MyMoney.Loss_Count+= 1
                MyMoney.buy_price, MyMoney.log_buy = 0,0     
            if  (update_data.iloc[i-2]['CCI']<100 < update_data.iloc[i]['CCI'] or update_data.iloc[i-2]['RSI']<80 < update_data.iloc[i]['RSI'] )and update_data.iloc[i-1]['in_uptrend10'] == update_data.iloc[i-1]['in_uptrend20'] == True and MyMoney.buy_price ==0 :
                    if MyMoney.wallet >= 10000 :
                        MyMoney.start_cap = MyMoney.wallet - 9000
                    else :
                        MyMoney.start_cap = 1000
                    MyMoney.wallet = buy(MyMoney.wallet,MyMoney.start_cap)
                    MyMoney.buy_price = update_data.iloc[i]['Open']
                    MyMoney.stoploss = 0.8*MyMoney.buy_price
                    MyMoney.log_buy = i
            
        if(MyMoney.buy_price > 0):
            MyMoney.wallet = MyMoney.wallet + MyMoney.start_cap
            MyMoney.buy_price = 0
        print(MyMoney.wallet,'-------',a,'--win: ',MyMoney.Win_Count,'--loss: ',MyMoney.Loss_Count,'---',update_data.Volume[len(update_data)-1])
        avg_profit.append(MyMoney.wallet)
    print (np.mean(avg_profit))

while True:
    # cap10_main()
    #print (client.get_symbol_ticker(symbol = symbol )['price'])
    #print (coinname)
    print('--------------------------------------------------------------')
    #sleep(1)
    # printe("Test")

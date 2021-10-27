from datetime import date
from matplotlib import colors, ticker, transforms
from matplotlib import lines
import pandas
import pandas as pd
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
def buy(curr_wallet):
    curr = curr_wallet - 1000
    return curr
def sell(buy_price, curr_price, curr_wallet):
    a= curr_price*1000/buy_price + curr_wallet
    return a
def getminutedata(symbol,inter,lookback):   
    frame = pd.DataFrame(client.get_historical_klines(symbol,inter,lookback))
    frame =frame.iloc[:,:6]
    frame.columns = ['Time','Open','High','Low','Close','Column']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit ='ms')
    frame = frame.astype(float)
    return frame

def Crawl_Data():
    #data = getminutedata('BTCUSDT',time_interval+'m',str(his)+' min ago UTC')
    #data = getminutedata('BTCUSDT',Client.KLINE_INTERVAL_4HOUR,'4004 hours ago UTC')
    data = getminutedata('BTCUSDT',Client.KLINE_INTERVAL_1HOUR,'1002 hours ago UTC')
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
    ADX(data,5)
    
    #    
    #---------------------------------------------------------------------SMA & EMA --------------------------------------------------------------------------------------
    def MA(df,number):
        df['MA'+str(number)] = df['Close'].rolling(number).mean()
        return df
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA200'] = data['Close'].rolling(200).mean()
    MA(data,30)
    def EMA(df,number):
        data['EMA'+str(number)] = data['Close'].ewm(span=number,adjust=False).mean()
    EMA(data,20)
    #data['EMA100'] = data['Close'].ewm(span=100,adjust=False).mean()
    #print(data.tail(50))
    #---------------------------------------------------------------------- MACD --------------------------------------------------------------------------------------
    def MACD(df):
        exp1 = df['Close'].ewm(span= 12 , adjust= False).mean()
        exp2 = df['Close'].ewm(span= 26 , adjust= False).mean()
        df['MACD']=exp1 -exp2
        df['Signal']  = df['MACD'].ewm(span= 9 , adjust= False).mean()
        return df
    MACD(data)

    #---------------------------------------------------------------------- Stochastic Oscillator ------------------------------------------------------------------------
    def Stochartic(df):
        high14 = df['High'].rolling(14).max()
        low14 = df['Low'].rolling(14).min()
        df['%K'] = (df['Close']-low14)*100/(high14-low14)
        df['%D'] = df['%K'].rolling(3).mean()
        return df
    Stochartic(data)
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
    ichimoku(data)

#---------------------------------------------------------------------- Bollinger Bands ------------------------------------------------------------------------------------------
    def bollinger_bands(df):
        SMA = df['Close'].rolling(window= 20).mean()
        stddev = df['Close'].rolling(window= 20).std()
        df['Upper'] = SMA + 2*stddev
        df['Low'] = SMA - 2*stddev
        return df
    bollinger_bands(data)

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


#------------------------------------------------------------Support Resistance------------------- Chưa Viết Xong---------------------------------

    # def suport_resistance(df):
    #     def support(df1,l,n1,n2):
    #         for i in range(l-n1+1,l+1):
    #             if(df1.Low[i]> df1.Low[i-1]):
    #                 return 0
    #         for i in range(l+1,l+n2+1):
    #             if(df1.Low[i]< df1.Low[i-1]):
    #                 return 0
    #         return 1
    #     # support (df,46,3,2)
    #     def resistance(df1,l,n1,n2):
    #         for i in range(l-n1+1,l+1):
    #             if(df1.High[i]< df1.High[i-1]):
    #                 return 0
    #         for i in range(l+1,l+n2+1):
    #             if(df1.High[i]> df1.High[i-1]):
    #                 return 0
    #         return 1
    
    data['Index']= range(len(data))

    data = data.dropna()   # Lọc những dữ liệu NaN
    data['price change'] = data['Close'].pct_change()
    data = data.dropna()   # Lọc những dữ liệu NaN
   # print(data.tail(50))


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
# def ADX_graph(df):
   
#     df=df.astype(np.float64)
#     df[['+DI','-DI','ADX']].plot()
#     plt.draw()
#     plt.xlabel('Date',fontsize=18)
#     plt.ylabel('Close Pride',fontsize=18)    
#     plt.pause(60)
#     plt.clf()
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
    
    #update_data= []
    update_data = Crawl_Data()
    #print(update_data.head(50))
    plt.cla() 
    plt.plot(update_data.index,update_data.EMA20,linewidth=1.0,color='green')
    plt.plot(update_data.index,update_data.Close,linewidth=1.0,color='green')
    #plt.plot(x_values,din,linewidth=1.0,color='blue')
    for i in range(90,len(update_data)):       
        #print (update_data.iloc[i-1]['ADX'], update_data.iloc[i-1]['MA20'] , update_data.iloc[i-1]['Low'] , MyMoney.buy_price ) 
        if  update_data.iloc[i-1]['%K'] > 80 and update_data.iloc[i-1]['MA30'] < update_data.iloc[i-1]['Close'] and update_data.iloc[i-1]['RSI'] >50 and MyMoney.buy_price == 0 :
                MyMoney.wallet = buy(MyMoney.wallet)
                MyMoney.buy_price = update_data.iloc[i]['Open']
                MyMoney.stoploss = 0.98*MyMoney.buy_price
                MyMoney.log_buy = i
                print('---------------')   
                
        if  (update_data.iloc[i-2]['CCI'] >100 > update_data.iloc[i-1]['CCI'] or update_data.iloc[i-2]['CCI'] >-0 > update_data.iloc[i-1]['CCI']) and  MyMoney.buy_price >0 :
            MyMoney.wallet = sell(MyMoney.buy_price,update_data.Open[i],MyMoney.wallet)
            if update_data.Open[i] >=  MyMoney.buy_price : MyMoney.Win_Count+= 1
            if update_data.Open[i] <  MyMoney.buy_price : MyMoney.Loss_Count+= 1
            print(MyMoney.buy_price ,'-----', update_data.Open[i],'-----',MyMoney.log_buy,'-----', i, '--Win--',MyMoney.Win_Count,'--Loss--',MyMoney.Loss_Count,'-----',abs(update_data.Open[i]-MyMoney.buy_price),'------',MyMoney.wallet)
            MyMoney.buy_price, MyMoney.log_buy = 0,0                    


    # for i in range(27,len(update_data)):
    #     suport_resistance(update_data)

#    din = update_data['-DI']
    #dinema = update_data['EMa5Din']
 #   plt.cla() 
   # plt.plot(x_values,dinema,linewidth=1.0,color='green')
 #   plt.plot(x_values,din,linewidth=1.0,color='blue')
    #plt.plot(x_values,y2_values,linewidth=1.0)
    # print(update_data.tail(50))
    # print(len(update_data))

    # for i in range(27,len(update_data)):    # Tìm điểm cắt nhau 
      
    #     a0,a1= update_data.iloc[i-1]['MACD'],update_data.iloc[i]['MACD']
    #     b0,b1= update_data.iloc[i-1]['Signal'],update_data.iloc[i]['Signal']
    #     dip0,dip1= update_data.iloc[i-1]['+DI'],update_data.iloc[i]['+DI']
    #     din0,din1= update_data.iloc[i-1]['+DI'],update_data.iloc[i]['+DI']
    #     av6_Adx = [update_data.iloc[i]['ADX'],update_data.iloc[i-1]['ADX'],update_data.iloc[i-2]['ADX'],update_data.iloc[i-3]['ADX'],update_data.iloc[i-4]['ADX'],update_data.iloc[i-5]['ADX']]
    #     #av6_Adx = [update_data.iloc[i]['ADX'],update_data.iloc[i-1]['ADX'],update_data.iloc[i-2]['ADX']]
    #     min_Adx = min(av6_Adx)
    #     max_Adx = max(av6_Adx)
    #     av_Adx = np.mean(av6_Adx)
    #     av6_Din = [update_data.iloc[i]['-DI'],update_data.iloc[i-1]['-DI'],update_data.iloc[i-2]['-DI'],update_data.iloc[i-3]['-DI'],update_data.iloc[i-4]['-DI'],update_data.iloc[i-5]['-DI']]
    #     #av6_Dinar =np.array([update_data.iloc[i-5]['Index'],update_data.iloc[i-4]['Index'],update_data.iloc[i-3]['Index'],update_data.iloc[i-2]['Index'],update_data.iloc[i-1]['Index'],update_data.iloc[i]['Index']], [update_data.iloc[i-5]['-DI'],update_data.iloc[i-4]['-DI'],update_data.iloc[i-3]['-DI'],update_data.iloc[i-2]['-DI'],update_data.iloc[i-1]['-DI'],update_data.iloc[i]['-DI']])
    #     #em6_Din = av6_Dinar[1].ewm(span=6,adjust=False).mean()
    #     #av6_Din = [update_data.iloc[i]['-DI'],update_data.iloc[i-1]['-DI'],update_data.iloc[i-2]['-DI']]
    #     min_Din = min(av6_Din)
    #     max_Din = max(av6_Din)
    #     av_Din = np.mean(av6_Din)
    #     av6_Dip = [update_data.iloc[i]['+DI'],update_data.iloc[i-1]['+DI'],update_data.iloc[i-2]['+DI'],update_data.iloc[i-3]['+DI'],update_data.iloc[i-4]['+DI'],update_data.iloc[i-5]['+DI']]
    #     min_Dip = min(av6_Dip)
    #     max_Dip = max(av6_Dip)
    #     av_Dip = np.mean(av6_Dip)
    #     cci0,cci1 = update_data.iloc[i-1]['CCI'],update_data.iloc[i]['CCI']

    #     # Theo chi so nen sinh doi

    #     #if ((update_data.iloc[i-2]['Open'] - update_data.iloc[i-2]['Close']) > 0) and ((update_data.iloc[i-1]['Close'] - update_data.iloc[i-1]['Open']) > 0) and (abs(update_data.iloc[i-1]['Close'] - update_data.iloc[i-1]['Open']) - abs(update_data.iloc[i-2]['Open'] - update_data.iloc[i-2]['Close'])  > 0 ) and MyMoney.buy_price == 0 :
    #     # Fail Win 35 Loss 43 -> 9900
    #     if update_data.iloc[i]['High'] > 1.03*MyMoney.buy_price and MyMoney.buy_price > 0:
    #         MyMoney.wallet = sell(MyMoney.buy_price,1.03*MyMoney.buy_price,MyMoney.wallet)
    #         MyMoney.Win_Count+= 1
    #         print(MyMoney.buy_price ,'-----', 1.03*MyMoney.buy_price,'--------------',MyMoney.log_buy,'--------------------------', i, '----Win-----',MyMoney.Win_Count,'-----',MyMoney.wallet)
    #         MyMoney.buy_price, MyMoney.log_buy = 0,0

        
    #     if update_data.iloc[i]['Low'] <= MyMoney.stoploss and MyMoney.buy_price > 0 :
    #         MyMoney.wallet = sell(MyMoney.buy_price,MyMoney.stoploss,MyMoney.wallet)
    #         MyMoney.Loss_Count+= 1
    #         print(MyMoney.buy_price ,'-----', MyMoney.stoploss,'--------------',MyMoney.log_buy,'--------------------------', i, '----Loss-----',MyMoney.Loss_Count,'-----',MyMoney.wallet)
    #         MyMoney.buy_price, MyMoney.log_buy = 0,0   

    #     if ((update_data.iloc[i-2]['Open'] - update_data.iloc[i-2]['Close']) > 0) and ((update_data.iloc[i-1]['Close'] - update_data.iloc[i-1]['Open']) > 0) and (abs(update_data.iloc[i-1]['Close'] - update_data.iloc[i-1]['Open']) - abs(update_data.iloc[i-2]['Open'] - update_data.iloc[i-2]['Close'])  > 0) and MyMoney.buy_price == 0 :
             
    #             MyMoney.wallet = buy(MyMoney.wallet)
    #             MyMoney.buy_price = update_data.iloc[i]['Open']
    #             MyMoney.stoploss = 0.97*MyMoney.buy_price
    #             MyMoney.log_buy = i
    #             print('---------------')              
    #     # RSI > 50 ; DI+ > DI- ; CCI -2 < CCI - 1 & CCI -1 < 200 Buy(CCi i.open) ; %K > %D & %K > 80 ; ADX >25
    #     # if (update_data.iloc[i-2]['CCI'] < update_data.iloc[i-1]['CCI']) and update_data.iloc[i-1]['CCI'] < 200 and MyMoney.buy_price == 0 :
    #     #     if (update_data.iloc[i-1]['ADX']>=25 and update_data.iloc[i-1]['+DI']>update_data.iloc[i-1]['-DI'] and 
    #     #         update_data.iloc[i-1]['%K']>update_data.iloc[i-1]['%D'] and update_data.iloc[i-1]['%K']>80 and 
    #     #         update_data.iloc[i-1]['RSI']>50) and update_data.iloc[i-1]['MACD']>update_data.iloc[i-1]['Signal']  :
    #     #         MyMoney.wallet = buy(MyMoney.wallet)
    #     #         MyMoney.buy_price = update_data.iloc[i]['Close']
    #     #         MyMoney.log_buy = i
    #     #         print('---------------') 

    #     #if  update_data.iloc[i]['CCI'] > update_data.iloc[i-1]['CCI'] and update_data.iloc[i-1]['CCI'] < -100 and MyMoney.buy_price == 0 :
    #     # if   update_data.iloc[i]['CCI'] >= -100 and update_data.iloc[i-1]['CCI'] < -100 and MyMoney.buy_price == 0 :    #update_data.iloc[i]['CCI'] > update_data.iloc[i-1]['CCI'] and update_data.iloc[i-1]['CCI'] < -100
    #     #     #if update_data.iloc[i]['-DI'] > update_data.iloc[i]['+DI'] and ((av_Din < update_data.iloc[i]['-DI'] and av_Adx > update_data.iloc[i]['ADX']) ) : # or (av_Din < update_data.iloc[i]['-DI'] and av_Adx > update_data.iloc[i]['ADX'])) :
    #     #     if  (update_data.iloc[i]['RSI']<40 or update_data.iloc[i-1]['RSI']<40 ) and (update_data.iloc[i]['RSI'] > update_data.iloc[i-1]['RSI'] )  and  update_data.iloc[i]['EMA100'] > update_data.iloc[i-5]['EMA100'] : # and update_data.iloc[i]['ADX'] > 25 and abs(update_data.iloc[i]['+DI'] - update_data.iloc[i]['-DI']) > 20 and ( (update_data.iloc[i]['EMa5Din'] > update_data.iloc[i-6]['EMa5Din'] and update_data.iloc[i]['ADX'] < update_data.iloc[i-6]['ADX'] ))
    #     #         #and (update_data.iloc[i]['RSI']<30 or update_data.iloc[i-1]['RSI']<30)  and and abs(update_data.iloc[i]['+DI'] - update_data.iloc[i]['-DI']) > 20 
    #     #         MyMoney.wallet = buy(MyMoney.wallet)
    #     #         MyMoney.buy_price = update_data.iloc[i]['Close']
    #     #         MyMoney.log_buy = i
    #     #         print('--------------------------') 
    #     #     # if update_data.iloc[i]['-DI'] < update_data.iloc[i]['+DI'] and ((av_Dip > update_data.iloc[i]['+DI'] and av_Adx < update_data.iloc[i]['ADX']) or (av_Dip < update_data.iloc[i]['+DI'] and av_Adx > update_data.iloc[i]['ADX'])) :
    #     #     #     MyMoney.wallet = buy(MyMoney.wallet)
    #     #     #     MyMoney.buy_price = update_data.iloc[i]['Close']
    #     #     #     MyMoney.log_buy = i
    #     #     #     print('---------------') 
    #     # if  ((update_data.iloc[i]['CCI']<=-20 and update_data.iloc[i-1]['CCI']>-20) or (update_data.iloc[i]['CCI']<=100 and update_data.iloc[i-1]['CCI']>100) or (update_data.iloc[i]['CCI']<=-100 and update_data.iloc[i-1]['CCI']>-100)) and MyMoney.buy_price > 0 : #or (update_data.iloc[i]['CCI']<=0 and update_data.iloc[i-1]['CCI']>0)  
    #     #     MyMoney.wallet = sell(MyMoney.buy_price,update_data.iloc[i]['Close'],MyMoney.wallet)
    #     #     print(MyMoney.buy_price ,'-----', update_data.iloc[i]['Close'],'--------------',MyMoney.log_buy,'-------------',i)
    #     #     MyMoney.buy_price, MyMoney.log_buy = 0,0 

    #     # a0,a1 = y_values[i-1],y_values[i]
    #     # b0,b1 = y2_values[i-1],y2_values[i]
    #     # if (a0>b0 and a1<b1) or (a0<b0 and a1>b1): 
    #     #     if  (a0<b0 and a1>b1):
    #     #         MyMoney.wallet = buy(MyMoney.wallet)
    #     #         MyMoney.buy_price = update_data.iloc[i]['Close']
    #     #        # print(MyClass.buy_price)
    #     #     if  (a0>b0 and a1<b1)and MyMoney.buy_price > 0:
    #     #         MyMoney.wallet = sell(MyMoney.buy_price,update_data.iloc[i]['Close'],MyMoney.wallet)
    #     #         print(MyMoney.buy_price ,'-----', update_data.iloc[i]['Close'],'--------------',i)

    #     #     x0,y0 = pt_AB(update_data.iloc[i-1]['Index'],a0,update_data.iloc[i]['Index'],a1)
    #     #     x1,y1 = pt_AB(update_data.iloc[i-1]['Index'],b0,update_data.iloc[i]['Index'],b1)
    #     #     M,N = pt_intersection(x0,y0,x1,y1)
    #     #     plt.plot(M, N, 'ro')

    #         ## Chỉ Số Main = CCI
    #     # if (update_data.iloc[i-1]['CCI']<-100 and update_data.iloc[i]['CCI'] >= -100)   and update_data.iloc[i]['ADX'] >= 25 and MyMoney.buy_price == 0  and update_data.iloc[i]['%K'] > update_data.iloc[i]['%D']  and update_data.iloc[i]['MACD'] > update_data.iloc[i]['Signal']: # and update_data.iloc[i]['+DI'] > update_data.iloc[i]['-DI']: #and update_data.iloc[i]['RSI'] > 60: #and update_data.iloc[i]['+DI'] > update_data.iloc[i]['-DI'] : and update_data.iloc[i]['MA5'] > update_data.iloc[i]['EMA100']
    #     #     MyMoney.wallet = buy(MyMoney.wallet)
    #     #     MyMoney.buy_price = update_data.iloc[i]['Close']
    #     #     MyMoney.log_buy = i
    #     #     print('---------------')
    #     # if (update_data.iloc[i]['Open'] < 0.9*MyMoney.buy_price  or (update_data.iloc[i]['CCI']<=100 and update_data.iloc[i-1]['CCI']>100)  ) and MyMoney.buy_price >0 :
    #     #     MyMoney.wallet = sell(MyMoney.buy_price,update_data.iloc[i]['Open'],MyMoney.wallet)
    #     #     print(MyMoney.buy_price ,'-----', update_data.iloc[i]['Open'],'--------------',MyMoney.log_buy,'--------------------------', i)
    #     #     MyMoney.buy_price, MyMoney.log_buy = 0,0
           
    #     # if  update_data.iloc[i]['CCI'] < 100 and update_data.iloc[i]['MACD'] > 0 and update_data.iloc[i]['MACD'] > update_data.iloc[i]['Signal'] and update_data.iloc[i]['+DI'] > update_data.iloc[i]['-DI'] and update_data.iloc[i]['ADX'] >= 25 and MyMoney.buy_price == 0: #and MyMoney.buy_price == 0 and update_data.iloc[i]['%K'] > update_data.iloc[i]['%D'] and update_data.iloc[i]['+DI'] > update_data.iloc[i]['-DI'] :
    #     #         MyMoney.wallet = buy(MyMoney.wallet)
    #     #         MyMoney.buy_price = update_data.iloc[i]['Close']
    #     #         MyMoney.log_buy = i
    #     #         print('---------------')               
    #     # if  ((a0>b0 and a1<b1) or update_data.iloc[i]['MACD'] < update_data.iloc[i]['Signal'] < 0 or update_data.iloc[i]['%K'] < update_data.iloc[i]['%D']) and MyMoney.buy_price > 0:
    #     #     MyMoney.wallet = sell(MyMoney.buy_price,update_data.iloc[i]['Open'],MyMoney.wallet)
    #     #     print(MyMoney.buy_price ,'-----', update_data.iloc[i]['Open'],'--------------',MyMoney.log_buy,'--------------------------', i)
    #     #     MyMoney.buy_price, MyMoney.log_buy = 0,0        

    #     #     x0,y0 = pt_AB(update_data.iloc[i-1]['Index'],a0,update_data.iloc[i]['Index'],a1)
    #     #     x1,y1 = pt_AB(update_data.iloc[i-1]['Index'],b0,update_data.iloc[i]['Index'],b1)
    #     #     M,N = pt_intersection(x0,y0,x1,y1)
    #     #     plt.plot(M, N, 'ro')
        
    if(MyMoney.buy_price > 0):
        MyMoney.wallet = MyMoney.wallet + 1000
        MyMoney.buy_price = 0
    print(MyMoney.wallet)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Name')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(),main,interval=10000) 

        
plt.tight_layout()
plt.show()
    

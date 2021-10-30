import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import stockdata, corona
import sys
import pandas
from statsmodels.tsa.arima_model import ARIMA


#"Malgun Gothic" 폰트 설정
sns.set(font="Malgun Gothic",
        rc={"axes.unicode_minus":False},
        style='darkgrid')

if __name__ == "__main__":
    app = stockdata.QApplication(sys.argv)
    trade = stockdata.system_trading()
    '''
    # stock1(생물공학 - 알테오젠)
    trade.rq_chart_data("196170", "20211016", 1)
    stock1_day_data = pandas.DataFrame(trade.day_data, columns=['date','open','high','low','close','volume','trade_volume'])
    # stock3(생명과학도구및서비스 - 씨젠)
    trade.rq_chart_data("096530", "20211016", 1)
    stock3_day_data = pandas.DataFrame(trade.day_data, columns=['date','open','high','low','close','volume','trade_volume'])
    '''
    # stock2(해운사 - hmm)
    trade.rq_chart_data("011200", "20211016", 1)
    stock2_day_data = pandas.DataFrame(trade.day_data, columns=['date','open','high','low','close','volume','trade_volume'])


    #dtype object->int형으로 변환
    '''
    stock1_day_data = stock1_day_data.apply(pd.to_numeric)
    df1 = stock1_day_data.sort_values(by='date', ascending=True)
    stock3_day_data = stock3_day_data.apply(pd.to_numeric)
    df3 = stock3_day_data.sort_values(by='date', ascending=True)
    '''
    stock2_day_data = stock2_day_data.apply(pd.to_numeric)
    df2 = stock2_day_data.sort_values(by='date', ascending=True)

    #주식데이터 df1,df2,df3와 코로나데이터 corona.df 합치기
    '''
    df1_inner = pandas.merge(df1, corona.df, on='date', how='inner')
    df3_inner = pandas.merge(df3, corona.df, on='date', how='inner')
    '''
    df2_inner = pandas.merge(df2, corona.df, on='date', how='inner')
    #종가, 확진자수, 일일확진자수, 1차접종완료, 접종완료 컬럼추출
    # train_df1 = df1_inner[['close', '확진자수', '일일확진자수', '1차접종완료', '접종완료']]
    train_df2 = df2_inner[['close', '확진자수', '일일확진자수', '1차접종완료', '접종완료']]
    # train_df3 = df3_inner[['close', '확진자수', '일일확진자수', '1차접종완료', '접종완료']]

    #상관계수
    # df1_corr = train_df1.corr(method='pearson')
    df2_corr = train_df2.corr(method='pearson')
    # df3_corr = train_df3.corr(method='pearson')

    # print(df1_corr)
    print(df2_corr)
    # print(df3_corr)
    #상관계수 히트맵
    '''
    plt.rcParams['figure.figsize'] = [10,10]
    sns.heatmap(train_df1.corr(), annot=True, cmap='Blues', vmin=-1, vmax=1)
    plt.title('생물공학 - 알테오젠', fontsize=15)
    plt.show()
    plt.rcParams['figure.figsize'] = [10, 10]
    sns.heatmap(train_df3.corr(), annot=True, cmap='Greens', vmin=-1, vmax=1)
    plt.title('생명과학도구및서비스 - 씨젠', fontsize=15)
    plt.show()
    '''
    # plt.rcParams['figure.figsize'] = [10, 10]
    # sns.heatmap(train_df2.corr(), annot=True, cmap='Pastel1', vmin=-1, vmax=1)
    # plt.title('해운사 - hmm', fontsize=15)
    # plt.show()

    #예측하기
    '''
    order=(2,1,2)
      2 : AR 관련데이터. 2번째 과거까지
      1 : 차분(Difference). 현재상태-바로이전상태의 차.
          시계열데이터의 불규칙성을 보정.
      2 : MA. 2번째 과거 정보의 오차를 이용해서 현재 추론.
    '''
    model = ARIMA(df2.close.values, order=(2, 1, 2))
    # 학습하기
    model_fit = model.fit(trend='c', full_output=True, disp=True)
    fig = model_fit.plot_predict()

    # resid : 잔차정보.
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()  # 두번째 그래프
    # 실제 데이터와의 비교
    # 예측 데이터
    forecast_data = model_fit.forecast(steps=5)  # 5일 정보를 예측
    print(forecast_data)  # 예측 데이터`

    # '''
    #    1번배열 : 예측값. 5일치 예측값
    #    2번배열 : 표준오차.
    #    3번배열 : [예측하한값, 예측상한값]
    # '''
    # #실데이터 읽기
    # test_file_path = 'market-price-test.csv'
    # bitcoin_test_df = \
    #     pd.read_csv(test_file_path, names=['ds', 'y'],header=0)
    # bitcoin_test_df
    # #5일치 예측 데이터.
    # pred_y = forecast_data[0].tolist()
    # #실제 데이터
    # test_y = bitcoin_test_df.y.values
    # pred_y_lower = [] #예측하한값들
    # pred_y_upper = [] #예측상한값들
    # for low_up in forecast_data[2] :
    #    pred_y_lower.append(low_up[0]) #하한값
    #    pred_y_upper.append(low_up[1]) #상한값
    # #시각화
    # plt.plot(pred_y, color="gold") #예측값
    # plt.plot(test_y, color="green") #실제값
    # plt.plot(pred_y_lower, color="red") #예측 하한값
    # plt.plot(pred_y_upper, color="blue") #예측 상한값
    #
    # # Facebook의 Prophet알고리즘을 활용하여 시계열 데이터 분석하기
    # # pip install pystan --upgrade
    # # Anaconda Prompt 창
    # # conda install -c conda-forge fbprophet
    # # pip install --upgrade pip
    # import pandas as pd
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from fbprophet import prophet #페이스북에서 구현한 모델
    #
    #
    # file_path="market-price.csv"
    # bitcoin_df = \
    #     pd.read_csv(file_path, names=['ds', 'y'],header=0)
    # bitcoin_df.head()
    #
    # prophet = prophet(seasonality_mode='multiplicative',\
    # yearly_seasonality=True,weekly_seasonality=True,\
    # daily_seasonality=False,changepoint_prior_scale=0.5)
    # # KeyError : metric_file  오류 발생.
    # # console restart  되는 오류 발생시
    # # pip uninstall pystan
    # # pip install pystan
    # prophet.fit(bitcoin_df) #학습하기
    # #5일 앞을 예측하기
    # future_data = prophet.make_future_dataframe\
    #     (periods=5, freq='d')
    # future_data
    # forecast_data = prophet.predict(future_data) #예측하기
    # forecast_data.shape
    # forecast_data.info()
    # '''
    #   ds : 날짜
    #   yhat : 예측값
    #   yhat_lower : 예측하한값
    #   yhat_upper : 예측상한값
    # '''
    # forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
    #
    # fig1 = prophet.plot(forecast_data)
    # fig2 = prophet.plot_components(forecast_data)
    #
    # y = bitcoin_df.y.values[5:] #첫 5일을 제외한 실데이터
    # y_pred = forecast_data.yhat.values[5:-5] #prophet모델에 의한 예측값
    #                                          #첫 5일 이후부터 마지막 5일 이전까지
    # y
    # y_pred
    #
    # ### 실 데이터와 비교하기
    # test_file_path = 'market-price-test.csv'
    # bitcoin_test_df = pd.read_csv(test_file_path,names=['ds','y'],header=0)
    # y = bitcoin_test_df.y.values
    # y_pred = forecast_data.yhat.values[-5:]
    # plt.plot(y_pred,color='gold')
    # plt.plot(y,color='green')
    #
    # pred_y = forecast_data.yhat.values[-5:]
    # pred_y_upper = forecast_data.yhat_upper.values[-5:]
    #
    # plt.plot(pred_y, color="gold") # 모델이 예상한 가격 그래프입니다.
    # plt.plot(pred_y_lower, color="red") # 모델이 예상한 최소가격 그래프입니다.
    # plt.plot(pred_y_upper, color="blue") # 모델이 예상한 최대가격 그래프입니다.
    # plt.plot(y, color="green") # 실제 가격 그래프입니다.




    # 뉴스 크롤링(네이버금융? 이베스트? - 일단 긍(1),부정(0)으로만 평가)

    # 추가 고려사항) 정해진 3종목 이외 다른 종목까지 진행?
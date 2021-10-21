# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:04:51 2021

@author: 송호현
"""
from fbprophet import Prophet # 페이스북에서 구현한 모델
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Alteogen.csv',header=0)
df.info()
df=df.astype({'date':'str'})
df=df[df['date'].str[:4] == '2021']
df=df.sort_values(by='date', ascending=True)
df.info()

alteogen_df=df[['date','close']]
alteogen_df.info()
alteogen_df.rename(columns={'date':'ds','close':'y'}, inplace=True)
alteogen_df.info()

prophet=Prophet(seasonality_mode='multiplicative',
               yearly_seasonality=True,
               weekly_seasonality=True,
               daily_seasonality=True,
               changepoint_prior_scale=0.5)

prophet.fit(alteogen_df) # 학습하기

# 30일 앞을 예측하기
future_month=prophet.make_future_dataframe(periods=30, freq='d')
future_month
forecast_month=prophet.predict(future_month) # 예측하기
'''
ds : 날짜
yhat : 예측값
yhat_lower : 예측 하한값
yhat_upper : 예측 상한값
'''
forecast_month.shape
forecast_month[['ds','yhat','yhat_lower','yhat_upper']].tail
forecast_month[-30:]

fig1=prophet.plot(forecast_month)
fig2=prophet.plot_components(forecast_month)


y=alteogen_df.y.values[-30:] # 첫 30일을 제외한 실데이터
y_pred=forecast_month.yhat.values[30:-30] # prophet 모델에 의한 예측값. 첫 30일 이후부터 마직막 30일 이전까지
y
y_pred

# 실데이터와 비교하기
alteogen_test_df=pd.read_excel('Alteogen_test.xlsx',names=['ds','y'],header=0)
y=alteogen_test_df.y.values
y_pred=forecast_month.yhat.values[-30:]
plt.plot(y_pred, color='gold')
plt.plot(y, color='green')
plt.show()

# 실데이터, 예측데이터, 예측하한가, 예측상한가 그래프로 작성하기
y_pred=forecast_month.yhat.values[-30:] # 예측데이터
yhat_lower=forecast_month.yhat_lower.values[-30:] # 예측하한가
yhat_upper=forecast_month.yhat_upper.values[-30:] # 예측상한가
plt.plot(y,color='green')
plt.plot(y_pred,color='gold')
plt.plot(yhat_lower, color='red')
plt.plot(yhat_upper, color='blue')
plt.show()

# 모델 평가하기
# r2-score, RMSE 값을 출력하기
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
r2=r2_score(y,y_pred)
r2
rmse=sqrt(mean_squared_error(y,y_pred))
rmse


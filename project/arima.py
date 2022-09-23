# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:52:18 2021

@author: 송호현
"""

# 시계열 데이터 분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

df = pd.DataFrame()
for page in range(3,21):
    url = 'http://finance.naver.com/item/sise_day.nhn?code=011200'
    url = '{url}&page={page}'.format(url=url, page=page)
    print(url)
    df = df.append(pd.read_html(requests.get(url, headers={'User-agent': 'Mozilla/5.0'}).text))
print(df.info())

df = df[['날짜', '종가']]
df = df.dropna()

hmm_df = df.rename(columns={'날짜': 'ds', '종가': 'y'})
hmm_df['y'] = hmm_df['y'].astype(int)
hmm_df['ds'] = pd.to_datetime(hmm_df['ds'])
hmm_df = hmm_df.sort_values(by=['ds'], ascending=True)
print(hmm_df)

# date 컬럼을 인덱스로 변경하기
df2 = hmm_df.set_index('ds')
hmm2_df=df2[['y']]
hmm2_df.info()

hmm2_df.plot()
plt.show()

from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(hmm2_df.y.values, order=(2,1,0))

# 학습하기
model_fit=model.fit(trend='c', full_output=True, disp=True)
print(model_fit.summary())
fig=model_fit.plot_predict() # 예측 결과를 그래프로 출력

# resid : 잔차정보
residuals=pd.DataFrame(model_fit.resid)
residuals.plot() # 두번째 그래프
plt.show()
# 예측데이터
forecast_month=model_fit.forecast(steps=5) # 5일 정보를 예측
forecast_month

'''
1번 배열 : 예측값. 5일치 예측값
2번 배열 : 표준 오차
3번 배열 : [예측 하한값, 예측 상한값]
'''
# 실데이터
# 5일치 예측데이터
test_y=hmm_df.y.values[-5:]
pred_y=forecast_month[0].tolist()
pred_y_lower=[] # 예측 하한값
pred_y_upper=[] # 예측 상한값
for low_up in forecast_month[2] :
    pred_y_lower.append(low_up[0])
    pred_y_upper.append(low_up[1])

plt.plot(pred_y, color="gold") # 예측값
plt.plot(test_y, color='green') # 실제값
plt.plot(pred_y_lower, color='red') # 예측 하한값
plt.plot(pred_y_upper, color='blue') # 예측 상한값
plt.show()

# r2-score, RMSE 값을 출력하기
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from math import sqrt
r2=r2_score(test_y, pred_y)
print("r2 :", r2)
rmse=sqrt(mean_squared_error(test_y, pred_y))
print("rmse :", rmse)
msle=mean_squared_log_error(test_y, pred_y)
print("mlse :", msle)
rmsle=msle**0.5
print("rmsle:", rmsle)

#분류모델로 등락예측
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import requests
from sklearn.model_selection import train_test_split

df = pd.DataFrame()
for page in range(1, 50):
    url = 'http://finance.naver.com/item/sise_day.nhn?code=011200'
    url = '{url}&page={page}'.format(url=url, page=page)
    print(url)
    df = df.append(pd.read_html(requests.get(url, headers={'User-agent': 'Mozilla/5.0'}).text))
print(df.info())

df = df[['날짜', '종가', '전일비', '시가', '고가', '저가', '거래량']]
df = df.dropna()

df.to_csv('DAYTON_hourly.csv')

hmm_df = df.rename(columns={'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'start',
                            '고가': 'high', '저가': 'low', '거래량': 'volume'})
hmm_df[['close', 'diff', 'start', 'high', 'low', 'volume']] = hmm_df[['close', 'diff', 'start', 'high', 'low', 'volume']].astype(int)
hmm_df['date'] = pd.to_datetime(hmm_df['date'])
hmm_df = hmm_df.sort_values(by=['date'], ascending=True)
hmm_df.set_index('date', inplace=True)
print(hmm_df)

hmm_df[['close']].plot(figsize=(20, 6))


# 오늘의 종가와 내일의 종가 비교
# fluctuation : 종가 상승, 하락 여부(오르거나 같으면1, 그외0)
def up_down(x):
    if x >= 0:
        return 1
    else:
        return 0


hmm_df['fluctuation'] = (hmm_df['close'].shift(-1)-hmm_df['close']).apply(up_down)
hmm_df.drop('diff', axis=1, inplace=True)
print(hmm_df.head())

# 7:3 train_test split
target = hmm_df['fluctuation']
stock_df = hmm_df.drop('fluctuation', axis=1)
train_X, test_X, train_y, test_y = train_test_split(stock_df, target, test_size=0.3, shuffle=False)
print(test_X.tail())

# kfold
from sklearn.model_selection import KFold
kfold = KFold(n_splits = 3, shuffle = False)



print(train_X.tail())
#모델링
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 교차검증(cross_validation)
# cross_val_score에서 분류모형의 scoring은 accuracy
from sklearn.model_selection import cross_val_score


# 분류모형
logistic = LogisticRegression()
knn = KNeighborsClassifier()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
naive = GaussianNB()
# SVM은 매개변수와 데이터 전처리 부분에서 신경써야함. 따라서 현재 사용x
# 추후 매겨변수를 선택하는 알고리즘을 짠 후 사용

models = [{'name': 'Logistic', 'model': logistic}, {'name': 'KNN', 'model': knn},
    {'name': 'DecisonTree', 'model': tree}, {'name': 'RandomForest', 'model': forest},
    {'name': 'NaiveBayes', 'model': naive}]

# # 교차 검증 accuracy
# def cv_accuracy(models):
#      for m in models:
#          print("Model {} CV score : {:.4f}".format(m['name'], np.mean(cross_val_score(m['model'],
#                         train_X, train_y, cv=kfold))))
# print(cv_accuracy(models))


from sklearn import datasets
from sklearn import metrics
#모델 정확도
for m in models:
    model = m['model']
    model.fit(train_X, train_y)
    predict = model.predict(test_X)
    # Accuracy : 전체 샘플 중 맞게 예측한 샘플수의 비율
    # Precision(정밀도) : postive라고 예측한 것 중에서 실제 postive인 것
    # Recall(재현율) : 실제 postive중에 예측한 postive 비율
    print('model name : {}'.format(m['name']))
    print(metrics.classification_report(test_y, predict))
    print('Confusion Matrix')
    print(metrics.confusion_matrix(test_y, predict))
    print('Accuracy Score : {:.4f}\n'.format(metrics.accuracy_score(test_y, predict)))

#수익률
def rate_of_return():
    df['percent'] = round((df.close-df.close.shift(1))/df.close.shift(1)*100, 2)
    for i in range(len(df)-1):
        if (df.loc[i, 'predict'] == 0):
            df.loc[i+1, 'percent'] = df.loc[i+1, 'percent']

#모델별 수익률
for m in models:
    model = m['model']
    model.fit(train_X, train_y)
    predict = model.predict(test_X)
    df = pd.concat([test_X.reset_index().drop('date', axis=1), pd.DataFrame(predict, columns=['predict'])], axis=1)
    rate_of_return()
    df.dropna(inplace=True)

    print('model name : {}'.format(m['name']))
    print('첫날을 제외한 거래일수 : {}'.format(len(df)))
    print('누적 수익률 : {}'.format(round(df['percent'].sum(), 2)))
    print('1일 평균 수익률 : {}\n'.format(round(df['percent'].sum() / (len(df) - 1), 2)))

print("==========================================================================")
# 주가 보조지표활용 등락 예측
stock_assist = stock_df.copy()
# 이동평균선
def ma(df):x
    df['MA_5'] = df['close'].rolling(window=5).mean()
    df['MA_20'] = df['close'].rolling(window=20).mean()
# 지수이동평균선
def ema(df, day):
    df['EWM_{}'.format(day)] = df['close'].ewm(span=day).mean()
# 이격도
def ppo(df, day):
    df['PPO_{}'.format(day)] = (df['close']/df['MA_{}'.format(day)])*100
# RSI
def U(x):
    if x >= 0:
        return x
    else:
        return 0
def D(x):
    if x <= 0:
        return x * (-1)
    else:
        return 0
def rsi(df):
    df['diff_rsi'] = (df['close'].shift(1) - df['close'])
    df['AU'] = df['diff_rsi'].apply(U).rolling(window=5).mean()
    df['AD'] = df['diff_rsi'].apply(D).rolling(window=5).mean()
    df['RSI'] = df['AU'] / (df['AU'] + df['AD'])
# # 모멘텀 스토캐스틱 %K, %D, Fast, Slow
#
# def high_low(day):
#     global stock_assist
#     stock_assist = stock_assist.reset_index()
#     # for i in range(len(stock_assist) - day + 1):
#     for i in range(486):
#         stock_assist.loc[i, 'high_st'] = stock_assist[i:i + day]['high'].max()
#         stock_assist.loc[i, 'low_st'] = stock_assist[i:i + day]['low'].min()
#         stock_assist['high_st_4'] = stock_assist['high_st'].shift(4)
#         stock_assist['low_st_4'] = stock_assist['low_st'].shift(4)
#         stock_assist['fast_K'] = (stock_assist['close'] - stock_assist['low_st_4']) / (stock_assist['high_st_4'] - stock_assist['low_st_4'])
#         stock_assist['fast_D'] = stock_assist['fast_K'].rolling(5).mean()
#         stock_assist['slow_K'] = stock_assist['fast_D']
#         stock_assist['slow_D'] = stock_assist['slow_K'].rolling(5).mean()
#         stock_assist = stock_assist.set_index('date')
# CCI
def CCI(df):
    #CCI = (M-N) / (0.015*D)
    # M = 특정일의 고가,저가, 종가의 평균
    # N = 일정기간동안의 단순이동평균 통상적으로 20일로 사용
    # D = M-N의 일정기간동안의 단순이동평균
    M = ((df.high)+(df.low)+(df.close)) / 3
    N = M.rolling(5).mean()
    D = (M-N).rolling(5).mean()
    CCI = (M - N)/ (0.015 * D)
    stock_assist['CCI'] = CCI
#macd
def macd(df, short=12, long=26, t=9):
    ma_12 = df.close.ewm(span=12).mean()
    ma_26 = df.close.ewm(span=26).mean()  # 장기(26) EMA
    macd = ma_12 - ma_26  # MACD
    macdSignal = macd.ewm(span=9).mean()  # Signal
    macdOscillator = macd - macdSignal  # Oscillator
    stock_assist['macd'] = macdOscillator

# 보조지표 추가
index = [ma, ema, ppo, rsi, CCI, macd]
for ex in index:
    if ex == ma:
        ex(stock_assist)
    if ex == ema:
        ex(stock_assist, 5)
        ex(stock_assist, 20)
    if ex == ppo:
        ex(stock_assist, 5)
        ex(stock_assist, 20)
    if ex == rsi:
        ex(stock_assist)
        stock_assist.drop(['diff_rsi', 'AU', 'AD'], axis=1, inplace=True)
    # if ex == high_low:
    #     ex(stock_assist)
    #     stock_assist.drop(['high_st', 'low_st', 'high_st_4', 'low_st_4', 'fast_K', 'fast_D'], axis=1)
    if ex == CCI:
        ex(stock_assist)
    if ex == macd:
        ex(stock_assist)
print(stock_df)
print(stock_assist)

stock_assist = pd.concat([stock_assist, target], axis=1)
stock_assist.dropna(inplace=True)
target = stock_assist['fluctuation']
stock_assist = stock_assist.drop('fluctuation', axis=1)


# 보조지표 활용 모델 정확도
train_X1, test_X1, train_y1, test_y1 = train_test_split(stock_assist, target, test_size=0.3, shuffle=False)
for m in models:
    model = m['model']
    model.fit(train_X1, train_y1)
    predict = model.predict(test_X1)
    # Accuracy : 전체 샘플 중 맞게 예측한 샘플수의 비율
    # Precision(정밀도) : postive라고 예측한 것 중에서 실제 postive인 것
    # Recall(재현율) : 실제 postive중에 예측한 postive 비율
    print('model name : {}'.format(m['name']))
    print(metrics.classification_report(test_y1, predict))
    print('Confusion Matrix')
    print(metrics.confusion_matrix(test_y1, predict))
    print('Accuracy Score : {:.4f}\n'.format(metrics.accuracy_score(test_y1, predict)))

# 보조지표 활용 모델별 수익률
for m in models:
    model = m['model']
    model.fit(train_X1, train_y1)
    predict = model.predict(test_X1)
    df = pd.concat([test_X1.reset_index().drop('date', axis=1), pd.DataFrame(predict, columns=['predict'])], axis=1)
    rate_of_return()
    df.dropna(inplace=True)

    print('model name : {}'.format(m['name']))
    print('첫날을 제외한 거래일수 : {}'.format(len(df)))
    print('누적 수익률 : {}'.format(round(df['percent'].sum(), 2)))
    print('1일 평균 수익률 : {}\n'.format(round(df['percent'].sum() / (len(df) - 1), 2)))

# p-value
from sklearn.preprocessing import MinMaxScaler
stock_data = stock_assist.copy()
stock_data['fluctuation'] = (stock_data['close'].shift(-1)-stock_data['close']).apply(up_down)
scaler = MinMaxScaler()
columns = [i for i in stock_data.columns]
stock_data[columns] = scaler.fit_transform(stock_data)

import statsmodels.api as sm
logis = sm.Logit(stock_data['fluctuation'],stock_data[stock_df.columns])
result=logis.fit()

print(result.summary())

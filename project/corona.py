import pandas as pd
import numpy as np
#import plotly.graph_objects as go
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/pc/Desktop/data/COVID19_data.csv", encoding='utf8',thousands=',')
df = df.dropna(axis=0)

#기준일 -> date
df['date'] = df['기준일'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df[['date', '1차접종완료', '접종완료']]=df[['date', '1차접종완료', '접종완료']].apply(pd.to_numeric)
df = df[df['접종완료'] != 0]
#print(df.info())
#print(df.head())

# fig = go.Figure()
# fig.add_trace(
#     go.Scatter(x=df['기준일'], y=df['확진자수'])
# )
# fig.update_layout(title='일별 확진자수')
# fig.show()


#
# fig = go.Figure(data=go.Scatter( # x축 값을 생략한 경우 DataFrame의 Index에 의해 배치됨
#     y = df['sepal_length'], # y축 값 sepal_length 값에 따라 배치
#     mode='markers', # Scatter Plot을 그리기 위해 Markers
#     marker=dict(    # Marker에 대한 세부적은 설정을 지정
#         size=20,    # 점 크기
#         color=df['petal_length'], # 색깔 값을 petal_length에 따라 변하도록 설정
#         colorscale='Viridis', # one of plotly colorscales
#         showscale=True,  # colorscales 보여줌
#         line_width=1, # 마커 라인 두께 설정
#     )
# ))
# fig.update_layout(title='Iris Data')
# fig.show()

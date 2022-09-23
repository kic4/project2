import pandas as pd

df = pd.read_csv('news.csv')
df = df.set_index('기사제목')
print(df)
df.to_csv('news.txt', mode='w', encoding='utf-8-sig')
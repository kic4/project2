from konlpy.tag import Okt
from collections import Counter
import re
import pandas as pd

f = open('news.txt', 'r', encoding="utf-8")
news = f.read()
news = re.sub('[^A-Za-zㄱ-ㅎㅏ-ㅣ가-힣0-9]', '', news)
print(news)
#형태소 분석
okt = Okt()
nouns = okt.nouns(news)

#단어 빈도수 카운트
count = Counter(nouns)
print(count)
#사진 가져오기
import numpy as np
from PIL import Image
coloring = np.array(Image.open("wordcloud_shipping.png"))

# #사진 색깔 뽑아오기
# from wordcloud import ImageColorGenerator
# image_colors = ImageColorGenerator(coloring)



#단어 구름 만들기
from wordcloud import WordCloud, STOPWORDS
stopword = set(STOPWORDS)
df = pd.read_csv('stopwords.csv', header=None, encoding='utf-8')
stopword=[]
for i in range(0,len(df)):
    word=df[0][i]
    stopword.append(word)

wordcloud = WordCloud(font_path="MaruBuri-Regular.ttf", mask=coloring, background_color='white', max_words=20, stopwords=stopword).generate_from_frequencies(count)

#이미지 띄우기
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15)) #10,10 하면 좀 작아서 깨지고 20,20은 너무 과하게 크더라. 추측이지만 cm인 것 같다.
plt.imshow(wordcloud.recolor(colormap='hsv'))
plt.axis('off')
plt.savefig('wordcloud_final.jpg')
plt.show()

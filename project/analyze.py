import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import stockdata, corona
import sys
import pandas
#"Malgun Gothic" 폰트 설정
sns.set(font="Malgun Gothic",
        rc={"axes.unicode_minus":False},
        style='darkgrid')

if __name__ == "__main__":
    app = stockdata.QApplication(sys.argv)
    trade = stockdata.system_trading()

    # # stock1(생물공학 - 알테오젠)
    # trade.rq_chart_data("196170", "20211016", 1)
    # stock1_day_data = pandas.DataFrame(trade.day_data, columns=['date','open','high','low','close','volume','trade_volume'])
    # stock2(해운사 - hmm)
    trade.rq_chart_data("011200", "20211016", 1)
    stock2_day_data = pandas.DataFrame(trade.day_data, columns=['date','open','high','low','close','volume','trade_volume'])
    # # stock3(생명과학도구및서비스 - 씨젠)
    # trade.rq_chart_data("096530", "20211016", 1)
    # stock3_day_data = pandas.DataFrame(trade.day_data, columns=['date','open','high','low','close','volume','trade_volume'])

    #dtype object->int형으로 변환
    # stock1_day_data = stock1_day_data.apply(pd.to_numeric)
    # df1 = stock1_day_data.sort_values(by='date', ascending=True)
    stock2_day_data = stock2_day_data.apply(pd.to_numeric)
    df2 = stock2_day_data.sort_values(by='date', ascending=True)
    # stock3_day_data = stock3_day_data.apply(pd.to_numeric)
    # df3 = stock3_day_data.sort_values(by='date', ascending=True)

    #주식데이터 df1,df2,df3와 코로나데이터 corona.df 합치기
    # df1_inner = pandas.merge(df1, corona.df, on='date', how='inner')
    df2_inner = pandas.merge(df2, corona.df, on='date', how='inner')
    # df3_inner = pandas.merge(df3, corona.df, on='date', how='inner')

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
    # plt.rcParams['figure.figsize'] = [10,10]
    #
    # sns.heatmap(train_df1.corr(), annot=True, cmap='Blues', vmin=-1, vmax=1)
    # plt.title('생물공학 - 알테오젠', fontsize=15)
    # plt.show()
    plt.rcParams['figure.figsize'] = [10, 10]
    sns.heatmap(train_df2.corr(), annot=True, cmap='Pastel1', vmin=-1, vmax=1)
    plt.title('해운사 - hmm', fontsize=15)
    plt.show()
    # plt.rcParams['figure.figsize'] = [10, 10]
    # sns.heatmap(train_df3.corr(), annot=True, cmap='Greens', vmin=-1, vmax=1)
    # plt.title('생명과학도구및서비스 - 씨젠', fontsize=15)
    # plt.show()


    # K-means, decision tree - 주가예측

    # 뉴스 크롤링(네이버금융? 이베스트? - 일단 긍(1),부정(0)으로만 평가)

    # 추가 고려사항) 정해진 3종목 이외 다른 종목까지 진행?
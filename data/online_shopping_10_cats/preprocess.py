import pandas as pd
df = pd.read_csv('online_shopping_10_cats.csv', encoding='utf-8')
# df.drop_duplicates(keep='first', inplace=True)  # 去重，只保留第一次出现的样本
df = df.sample(frac=1.0)  # 全部打乱
# cut_idx = int(round(0.1 * df.shape[0]))
df['text']=df['review']
del df['review']
del df['cat']
df_test, df_train ,df_valio= df.iloc[:1000], df.iloc[2000:], df.iloc[1000:2000]
#dataframe = pd.DataFrame(df_train)
df_valio.to_csv("df_valio.csv",sep=',')
df_test.to_csv("test.csv",sep=',')
df_train.to_csv("train.csv",sep=',')
print ( df_test.shape,df_valio.shape, df_train.shape) #(1000, 3) (1000, 3) (60774, 3)


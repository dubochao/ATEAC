import pandas as pd
import os
def start():
    neg_txt=os.listdir('./neg')
    pos_txt=os.listdir('./pos')
    path= './data.csv'
    fp=open(path,'w',encoding='utf8')
    text,index=[],[]
    for i in neg_txt:
        with open('./neg/'+i,'r',errors='ignore',encoding='utf8') as fp:
            text.append(fp.read().replace('\n',''))
            index.append(0)
    for j in pos_txt:
        with open('./pos/'+j,'r',errors='ignore',encoding='utf8') as fp:
            text.append(fp.read().replace('\n',''))
            index.append(1)
    dict_txt={'label':index,'text':text}
    df=pd.DataFrame(dict_txt)

    a= df['text'].str.len()
    b=a[a<=2].index
    df=df.drop(b)
    df.to_csv('data.csv',encoding='utf_8_sig')

    # df = pd.read_csv('data.csv', encoding='utf-8')
    # df.drop_duplicates(keep='first', inplace=True)  # 去重，只保留第一次出现的样本

    df = df.sample(frac=1.0)  # 全部打乱
    cut_idx = int(round(0.1 * df.shape[0]))
    df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    #dataframe = pd.DataFrame(df_train)
    df_test.to_csv("test.csv",sep=',',encoding='utf_8_sig')
    df_train.to_csv("train.csv",sep=',',encoding='utf_8_sig')
    print (   df_test.shape, df_train.shape) # (3184, 12) (318, 12) (2866, 12)


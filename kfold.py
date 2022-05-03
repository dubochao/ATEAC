import torch
import pandas as pd
def get_kfold_data(k, i, X, y):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = pd.concat((X[0:val_start], X[val_end:]), axis=0)
        y_train = pd.concat((y[0:val_start], y[val_end:]), axis=0)
    else:  # 若是最后一折交叉验证
        X_valid, y_valid = X[val_start:], y[val_start:]  # 若不能整除，将多的case放在最后一折里
        X_train = X[0:val_start]
        y_train = y[0:val_start]

    return X_train, y_train, X_valid, y_valid

if __name__ == '__main__':
    df = pd.read_csv('data/dev.csv', encoding='utf-8',sep=',')
    X_train,y_train=df['text'],df['label']
    data = get_kfold_data(5, 3, X_train, y_train)
    train={'label':data[1],'text':data[0]}
    test={'label':data[3],'text':data[2]}
    train=pd.DataFrame(train).reset_index()
    test = pd.DataFrame(test).reset_index()
    del train['index']
    del test['index']
    test.to_csv("data/test.csv", sep=',', encoding='utf_8_sig')
    train.to_csv("data/train.csv", sep=',', encoding='utf_8_sig')
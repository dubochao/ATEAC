# 基于双层可解释注意力机制的情感分析 
- Aspect-level sentiment analysis based on a two-layer interpretable attention mechanism
## Pytorch for code implementation
- [gitee代码]( https://gitee.com/dbzstp/ATEAC)
## Reference resources
* 1.https://github.com/Embedding/Chinese-Word-Vectors
* 2.http://www.nlpir.org/
* 3.https://www.ctrip.com/

## Requirements 需要配置环境
- There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.
* python3.8
* pytorch==1.8.0
* torchtext==0.9.0
* tensorboard==2.6.0
* jieba==0.42.1

## Word vector 词向量训练模型采用
https://github.com/Embedding/Chinese-Word-Vectors<br>
* （Here is the trained word word2vec）
## Analysis data  数据分析
- Analyze the length of the dataset
```
python3 analyze_data.py
```
## 数据集划分 
-  data 文件夹下包含所有数据集 其文件夹下 preprocess.py 是对数据集进行划分
```
python preprocess.py
```
## Start training  
```
python3 main.py
```

## Predict
- 预测数据库句子
- Predict sentences outside the database
```
python3 predict.py
```

## 训练日志可视化
- tensorboard具体操作方法参考 https://tensorflow.google.cn/tensorboard/get_started
- 本模型和其他对比模型的 训练日志保存到 runs 文件夹,要查看日志，只需cmd进入shopping/Chn文件夹下运行下运行 
```
tensorboard --logdir=runs
```
* ChnSentiCorp-Htl-unba-10000
![image](https://user-images.githubusercontent.com/62787127/165257025-047fc667-330f-437a-b5d5-c0321899dd65.png)
* online_shopping_10_cats
![image](https://user-images.githubusercontent.com/62787127/165260514-f73dd28e-e5ea-429f-9789-495f3b228404.png)

## 训练结果

https://www.kaggle.com/code/zhenhoblngjia/notebook4778ed91b4/edit/run/86791127

https://www.kaggle.com/code/cccdxz/notebook3f7115162b/notebook


# DAWCC Multi channel mixed attention CNN for affective analysis of Chinese text
## Pytorch for code implementation
## Reference resources
* 1.https://github.com/Embedding/Chinese-Word-Vectors
* 2.http://www.nlpir.org/
* 3.https://www.ctrip.com/

## Requirements
- There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.
* python3.8
* pytorch==1.8.0
* torchtext==0.9.0
* tensorboard==2.6.0
* jieba==0.42.1

## Word vector
https://github.com/Embedding/Chinese-Word-Vectors<br>
* （Here is the trained word word2vec）
## Analysis data
- Analyze the length of the dataset
```
python3 analyze_data.py
```
## Start training
```
python3 main.py -h
```

## Predict
- 预测数据库外的句子
- Predict sentences outside the database
```
python3 predict.py
```

## 准确率
-  CNN-multichannel 使用预训练的静态词向量微调词向量
```
python main.py -static=true -non-static=true -multichannel=true
```
## 训练 CNN 并将日志保存到 runs 文件夹,要查看日志，只需cmd下运行
tensorboard --logdir=runs 

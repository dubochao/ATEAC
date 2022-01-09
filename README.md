# DAWCC Multichannel attention word2vec-CNN for affective analysis of Chinese text
## Pytorch for code implementation
## Reference resources
* https://github.com/yoonkim/CNN_sentence
* https://github.com/dennybritz/cnn-text-classification-tf
* https://github.com/Shawn1993/cnn-text-classification-pytorch

## Dependency
* python3.8
* pytorch==1.8.0
* torchtext==0.9.0
* tensorboard==2.6.0
* jieba==0.42.1

## Word vector
https://github.com/Embedding/Chinese-Word-Vectors<br>
* （Here is the trained word word2vec）
## 用法
```
python3 preprocess.py
```
## 用法
```
python3 main.py -h
```

## 训练
```
python3 main.py
```

## 准确率
-  CNN-multichannel 使用预训练的静态词向量微调词向量
```
python main.py -static=true -non-static=true -multichannel=true
```
## 训练 CNN 并将日志保存到 runs 文件夹,要查看日志，只需cmd下运行
tensorboard --logdir=runs 

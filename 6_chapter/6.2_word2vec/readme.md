# 说明
使用word2vec的两种方法来构建word embedding，同时将embedding降维显示在图像上

一种是skip-gram
```
w2v_skip_gram.py
```
一种是CBOW
```
w2v_cbow.py
```
两种方法都使用负采样的方法计算loss

# 输入
经过分词的汉语文章

# 输出

每个分词 + 128 维的词向量

词向量降维,可视化图片

# 其他

数据处理的时候用到采样方法来进行高频噪声去除

原理见https://zhuanlan.zhihu.com/p/27296712

# 说明

将source_data中的原始语料使用结巴分词，然后以字为单位进行标注，

这里的jieba加载了事先准备的医学词典“DICT_NOW.csv”，里边含有医学名词以及标签

然后进行分词,

分完词之后进行标注，使用BIO标注法

同时按照2:2:11的比例放在dev：test：train 三部分

最终的输出是：

example.dev

example.test

example.train



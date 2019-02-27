# 说明
Stanfordnlp的实例代码


# ner.py

输入 new.txt

输出 

ner.txt

pos_tag.txt

分别进行命名实体识别和词性标注

# test.py
输出
```
/home/kuo/anaconda2/envs/py3/bin/python /home/kuo/NLP/NLP_basis/1_Stanford_NLP_example/test.py
['清华', '大学', '位于', '北京', '。']
[('清华', 'NR'), ('大学', 'NN'), ('位于', 'VV'), ('北京', 'NR'), ('。', 'PU')]
[('清华', 'ORGANIZATION'), ('大学', 'ORGANIZATION'), ('位于', 'O'), ('北京', 'STATE_OR_PROVINCE'), ('。', 'O')]
```

# 注意
Stanfordnlp 代码库运行比较慢
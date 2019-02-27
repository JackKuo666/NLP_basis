# 说明
通过hanlp和nltk的CRF语法解析算法进行语法树构建，并使用for循环进行遍历树，然后分词合并成短语，然后标记

# cfg.py
通过hanlp和nltk的CRF语法解析算法进行语法树构建

```
/home/kuo/anaconda2/envs/py3/bin/python /home/kuo/NLP/NLP_basis/4_chapter/4.4自定义语法与CFG/cfg.py
test_nltk_cfg_en
[Tree('S', [Tree('NP', ['Mary']), Tree('VP', [Tree('V', ['saw']), Tree('NP', ['Bob'])])])]
Tree [1]: (S (N 我们) (VP (V 尊敬) (N 老师)))
Draw tree with Display ...
[Tree('S', [Tree('N', ['我们']), Tree('VP', [Tree('V', ['尊敬']), Tree('N', ['老师'])])])]
```

# extract_nvp.py

通过hanlp和nltk的CRF语法解析算法进行语法树构建，并使用for循环进行遍历树，然后分词合并成短语，然后标记

输入：

text.txt

输出：

nvp.txt


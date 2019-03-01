# 1.原理理解
# 1.1 textrank原理见 Textrank.docx
## 1.2 代码见：
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals
import sys
from operator import itemgetter
from collections import defaultdict
import jieba.posseg
from .tfidf import KeywordExtractor
from .._compat import *


class UndirectWeightedGraph:                                             # 无向有权图的定义
    d = 0.85

    def __init__(self):                                                  # 无向有权图就是一个词典，词典的key是后续要添加的词，value是一个由（起始点，终止点，边的权重）构成的三元组
        self.graph = defaultdict(list)

    def addEdge(self, start, end, weight):
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))                     # 由于是无向，所以终点也可以作为key

    def rank(self):                                                      # 执行textrank算法是在rank函数中完成的
        ws = defaultdict(float)
        outSum = defaultdict(float)

        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items():                                # 初始化各个节点的权值 # 统计各个节点的出度的次数之和
            ws[n] = wsdef
            outSum[n] = sum((e[2] for e in out), 0.0)

        # this line for build stable iteration
        sorted_keys = sorted(self.graph.keys())
        for x in xrange(10):  # 10 iters                                 # 遍历若干次
            for n in sorted_keys:                                        # 遍历各个节点
                s = 0
                for e in self.graph[n]:                                  # 遍历节点的入度节点
                    s += e[2] / outSum[e[1]] * ws[e[1]]                  # 将这些入度节点贡献后的权值相加： 贡献率= 入度节点与节点n的共现次数 / 入度节点的所有出度的次数
                ws[n] = (1 - self.d) + self.d * s                        # 更新节点n的权值

        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])

        for w in itervalues(ws):                                         # 获取权值的最大值和最小值
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n, w in ws.items():                                          # 对权值进行归一化
            # to unify the weights, don't *100.
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws


class TextRank(KeywordExtractor):                             # textrank算法抽取关键词所定义的类

    def __init__(self):
        self.tokenizer = self.postokenizer = jieba.posseg.dt  # 分词函数和词性标注函数
        self.stop_words = self.STOP_WORDS.copy()              # 停用词表
        self.pos_filt = frozenset(('ns', 'n', 'vn', 'v'))     # 词性过滤集合 （地名，名词，动名词，动词）
        self.span = 5                                         # 窗口

    def pairfilter(self, wp):                                                # 过滤条件：词性在词性过滤集合中，并且词的长度大于等于2，并且词不是停用词
        return (wp.flag in self.pos_filt and len(wp.word.strip()) >= 2
                and wp.word.lower() not in self.stop_words)

    def textrank(self, sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False):
        """
        Extract keywords from sentence using TextRank algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v'].
                        if the POS of w is not in this list, it will be filtered.
            - withFlag: if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        self.pos_filt = frozenset(allowPOS)
        g = UndirectWeightedGraph()                             # 定义无向有权图
        cm = defaultdict(int)                                   # 定义共现词典
        words = tuple(self.tokenizer.cut(sentence))             # 分词
        for i, wp in enumerate(words):                          # 依次遍历每个词
            if self.pairfilter(wp):                             # 词i满足过滤条件
                for j in xrange(i + 1, i + self.span):          # 依次遍历词i之后窗口范围内的词
                    if j >= len(words):                         # 词j不能超过整个句子中分词的数量
                        break
                    if not self.pairfilter(words[j]):           # 如果词j不满足过滤条件，则跳过
                        continue
                    if allowPOS and withFlag:                   # 将词i和词j作为key，出现的次数作为value，添加到共现词典中
                        cm[(wp, words[j])] += 1
                    else:
                        cm[(wp.word, words[j].word)] += 1

        for terms, w in cm.items():                             # 依次遍历共现词典的每个元素，将词i，词j作为一条边起始点和终止点，出现的次数作为边的权重
            g.addEdge(terms[0], terms[1], w)
        nodes_rank = g.rank()                                   # 运行textrank算法
        if withWeight:                                          # 根据指标进行排序
            tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)

        if topK:                                                # 输出topK个词作为关键词
            return tags[:topK]
        else:
            return tags

    extract_tags = textrank

```
# 2.利用jieba的textrank来进行关键词提取
主函数是tex_tank.py

结果：
```python
/home/kuo/anaconda2/envs/py3/bin/python "/home/kuo/NLP/NLP_basis/3_ chapter/3.6TextRank算法原理介绍/tex_rank.py"

Building prefix dict from the default dictionary ...
keywords by textrank:
Dumping model to file cache /tmp/jieba.cache
Loading model cost 2.203 seconds.
Prefix dict has been built succesfully.
线程 进程 单位 基本 调度 分派 局部变量 堆栈 资源 程序执行
```

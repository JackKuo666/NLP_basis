#encoding=utf8
import jieba.posseg as pseg 
strings="是广泛使用的中文分词工具，具有以下特点："
words = pseg.cut(strings)

for word, flag in words:
    print('%s %s' % (word, flag))

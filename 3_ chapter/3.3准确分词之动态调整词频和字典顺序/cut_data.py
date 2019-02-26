#-*- coding=utf8 -*-
import jieba
import re
from tokenizer import cut_hanlp


#jieba.load_userdict("dict.txt")

# # 设置高词频：一个
# jieba.suggest_freq('台中',tune=True)

# 设置高词频：dict.txt中的每一行都设置一下
# fp=open("dict.txt",'r',encoding='utf8')
# for line in fp:
#     line=line.strip()
#     jieba.suggest_freq(line, tune=True)

# # 设置高词频：dict.txt中的每一行都设置一下快速方法
# [jieba.suggest_freq(line.strip(), tune=True) for line in open("dict.txt",'r',encoding='utf8')]


if __name__=="__main__":
    string="台中正确应该不会被切开。"

    words_jieba = " ".join(jieba.cut(string,HMM=False))

    words_hanlp=cut_hanlp(string)
    print("words_jieba:"+words_jieba,'\n',"words_hanlp:"+words_hanlp)
    
  

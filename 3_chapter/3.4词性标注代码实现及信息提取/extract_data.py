#-*- coding=utf8 -*-
import jieba
import re
from tokenizer import seg_sentences

fp=open("text.txt",'r',encoding='utf8')
fout=open("out.txt",'w',encoding='utf8')
for line in fp:
    line=line.strip()
    if len(line)>0:
        fout.write(' '.join(seg_sentences(line))+"\n")
fout.close()
if __name__=="__main__":
    pass
    
  

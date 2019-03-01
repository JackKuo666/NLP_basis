#encoding=utf8
import os

dict_file=open("/home/kuo/NLP/module"+os.sep+"hanlp"+os.sep+"data"+os.sep+"dictionary"+os.sep+"custom"+os.sep+"resume_nouns.txt",'r',encoding='utf8')
d={}


[d.update({line:len(line.split(" ")[0])}) for line in dict_file]
f=sorted(d.items(), key=lambda x:x[1], reverse=True)
dict_file=open("/home/kuo/NLP/module"+os.sep+"hanlp"+os.sep+"data"+os.sep+"dictionary"+os.sep+"custom"+os.sep+"resume_nouns1.txt",'w',encoding='utf8')
[dict_file.write(item[0]) for item in f]
dict_file.close()
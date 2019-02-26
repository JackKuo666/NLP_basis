#_*_ coding=utf8 _*_
import jieba
import re
from grammer.rules import grammer_parse
with open("text.txt", 'r',encoding= 'utf8') as fp:
    with open("out.txt", 'w', encoding='utf8') as fout:
        [grammer_parse(line.strip(), fout) for line in fp if len(line.strip())>0]

if __name__ == "__main__":
    pass

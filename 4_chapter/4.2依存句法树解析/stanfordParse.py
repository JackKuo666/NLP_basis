#encoding=utf8
from stanfordcorenlp import StanfordCoreNLP
from nltk import Tree, ProbabilisticTree
nlp = StanfordCoreNLP('/home/kuo/NLP/module/stanfordnlp', lang='zh')
import nltk,re
grammer = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammer) #生成规则
pattern=re.compile(u'[^a-zA-Z\u4E00-\u9FA5]')
pattern_del=re.compile('(\a-zA-Z0-9+)')
def _replace_c(text):
    intab = ",?!()"
    outtab = "，？！（）"    
    deltab = " \n<li>< li>+_-.><li \U0010fc01 _"
    trantab=text.maketrans(intab, outtab,deltab)
    return text.translate(trantab)
def parse_sentence(text):
    text=_replace_c(text)
    try:
        if len(text.strip())>6:
            return Tree.fromstring(nlp.parse(text.strip()))
    except:
        pass
def pos(text):
    text=_replace_c(text)
    if len(text.strip())>6:
        return nlp.pos_tag(text)
    else:
        return False
def denpency_parse(text):
    return nlp.dependency_parse(text)
from nltk.chunk.regexp import *

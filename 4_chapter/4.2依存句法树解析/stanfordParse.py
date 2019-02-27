#encoding=utf8
from stanfordcorenlp import StanfordCoreNLP
from nltk import Tree, ProbabilisticTree
nlp = StanfordCoreNLP('/home/kuo/NLP/module/stanfordnlp', lang='zh')
import nltk,re


grammer = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammer)                         #生成规则
pattern=re.compile(u'[^a-zA-Z\u4E00-\u9FA5]')
pattern_del=re.compile('(\a-zA-Z0-9+)')


def _replace_c(text):
    """
    将英文标点符号替换成中文标点符号，并去除html语言的一些标志等噪音
    :param text:
    :return:
    """
    intab = ",?!()"
    outtab = "，？！（）"
    deltab = " \n<li>< li>+_-.><li \U0010fc01 _"
    trantab=text.maketrans(intab, outtab,deltab)
    return text.translate(trantab)


def parse_sentence(text):
    text=_replace_c(text)          # 文本去噪
    try:
        if len(text.strip())>6:    # 判断，文本是否大于6个字，小于6个字的我们认为不是句子
            return Tree.fromstring(nlp.parse(text.strip()))        # nlp.parse(text.strip())：是将句子变成依存句法树  Tree.fromstring是将str类型的树转换成nltk的结构的树
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

#encoding=utf8
import os,gc,re,sys
from stanfordcorenlp import StanfordCoreNLP



stanford_nlp = StanfordCoreNLP("/home/kuo/NLP/module"+os.sep+'stanfordnlp', lang='zh')



def ner_stanford(raw_sentence,return_list=True):
    if len(raw_sentence.strip())>0:
        return stanford_nlp.ner(raw_sentence) if return_list else iter(stanford_nlp.ner(raw_sentence))

def cut_stanford(raw_sentence,return_list=True):
    if len(raw_sentence.strip())>0:
        return stanford_nlp.pos_tag(raw_sentence) if return_list else iter(stanford_nlp.pos_tag(raw_sentence))





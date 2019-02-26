#encoding=utf8
import os,gc,re,sys

from jpype import *
root_path="/home/kuo/NLP/module/hanlp"
djclass_path="-Djava.class.path="+root_path+os.sep+"hanlp-1.6.2.jar:"+root_path
startJVM(getDefaultJVMPath(),djclass_path,"-Xms1g","-Xmx1g")

Tokenizer = JClass('com.hankcs.hanlp.tokenizer.StandardTokenizer')

def to_string(sentence,return_generator=False):
    if return_generator:
        return (word_pos_item.toString().split('/') for word_pos_item in Tokenizer.segment(sentence))
    else:
        return " ".join([word_pos_item.toString().split('/')[0] for word_pos_item in Tokenizer.segment(sentence)])
                                                  # 这里的“”.split('/')可以将string拆分成list 如：'ssfa/fsss'.split('/') => ['ssfa', 'fsss']

def seg_sentences(sentence,with_filter=True,return_generator=False):
    segs=to_string(sentence,return_generator=return_generator)
    if with_filter:
        g = [word_pos_pair[0] for word_pos_pair in segs if len(word_pos_pair)==2 and word_pos_pair[0]!=' ' and word_pos_pair[1] not in drop_pos_set]
    else:
        g = [word_pos_pair[0] for word_pos_pair in segs if len(word_pos_pair)==2 and word_pos_pair[0]!=' ']
    return iter(g) if return_generator else g

def cut_hanlp(raw_sentence,return_list=True):
    if len(raw_sentence.strip())>0:
        return to_string(raw_sentence) if return_list else iter(to_string(raw_sentence))




#encoding=utf8
import json,re
import numpy 
import pandas as pd
import numpy  as np
import itertools
from itertools import chain
from tokenizer import seg_sentences
pattern=re.compile(u'[^a-zA-Z\u4E00-\u9FA5]')
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer 
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()        
        return json.JSONEncoder.default(self, obj)
def _replace_c(text):
    intab = ",?!"
    outtab = "，？！"    
    deltab = ")(+_-.>< "
    trantab=text.maketrans(intab, outtab,deltab)
    return text.translate(trantab)
def remove_phase(phase_list):
    remove_phase="aa,ab,abc,ad,ao,az,a写字楼,a区,a地块,a客户,a施工方,a系列,a项目,a系统"
    remove_phase_set=set(remove_phase.split(","))
    phase_list_set=set(phase_list)
    phase_list_set.difference(remove_phase_set)
    return list(phase_list_set)

def get_words(sentence):     

    segs = (word_pos_item.toString().split('/') for word_pos_item in StandardTokenizer.segment(sentence))
    segs = (tuple(word_pos_pair) for word_pos_pair in segs if len(word_pos_pair)==2)
    segs = ((word.strip(),pos) for word, pos in segs if pos  in keep_pos_set)
    segs = ((word, pos) for word, pos in segs if word and not pattern.search(word))
    result = ' '.join(w for w,pos in segs if len(w)>0)    
    return result
def get_words_no_space(sentence):       
    segs = (word_pos_item.toString().split('/') for word_pos_item in StandardTokenizer.segment(sentence))
    segs = (tuple(word_pos_pair) for word_pos_pair in segs if len(word_pos_pair)==2)
    segs = ((word.strip(),pos) for word, pos in segs if pos  in keep_pos_set)
    segs = ((word, pos) for word, pos in segs if word and not pattern.search(word))
    result = [w for w,pos in segs]  
    return result
 
def tokenize_raw(text):
    split_sen=(i.strip() for i in re.split('。|,|，|：|:|？|！|\t|\n',_replace_c(text)) if len(i.strip())>5)
    return [get_words(sentence) for sentence in split_sen]  


def tokenize(text):
    return [tok.strip() for tok in text.split(" ")]
def tokenize_no_space(text):
    return [get_words_no_space( text)]
def tokenize_triple(text):
    return "_".join([ tok for tok in text.split(" ")])
def pro(text):
    fout=open("triple.txt", "w", encoding='utf-8')
    vectorize=CountVectorizer(input='content', encoding='utf-8', decode_error='strict', 
                              strip_accents=None, lowercase=True, 
                   preprocessor=None, tokenizer=tokenize, 
                   stop_words=None, 
                   token_pattern=r"[a-zA-Z\u4E00-\u9FA5]", 
                   ngram_range=(3,3), analyzer='word', max_df=0.7, 
                   min_df=50, max_features=None, vocabulary=None, 
                   binary=False, dtype=np.int64)
    freq=vectorize.fit(text)
    vectorizer1=CountVectorizer(max_df=0.7, 
                                min_df=50,tokenizer=None)
    freq1=vectorizer1.fit_transform(('_'.join(i.split(" ")) for i in freq.vocabulary_.keys()))   
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值  
    word_freq=(freq1[:][i].sum() for i in range(freq1.shape[1]))
    tfidf=transformer.fit_transform(freq1)#第一个fit_transform是计算tf-idf，第二个                
    tfidf_sum=(tfidf[:][i].sum() for i in range(tfidf.shape[1]))
    tfidf_dic=vectorizer1.get_feature_names()
    
    dic_filter={}
    
    def _add(wq,tf,i):
        dic_filter[tfidf_dic[i]]=[wq,tf]
    for i,(word_freq_one,w_one) in enumerate(zip(word_freq,tfidf_sum)):
        _add(word_freq_one, w_one, i)
    sort_dic=dict(sorted(dic_filter.items(),key=lambda val:val[1],reverse=False))#,reverse=False为降序排列,返回list
    fout.write(json.dumps(sort_dic, ensure_ascii=False,cls=NumpyEncoder)+"\n")               
    fout.close()    
def gen_dic(in_path,save_path):
    fp=open(in_path,'r',encoding='utf-8')
    fout=open(save_path,'w',encoding='utf-8')
    
    copus=[list(json.loads(line).keys()) for line in fp]
    copus=[''.join(ph.split("_")) for phase in copus for ph in phase] 
    copus=remove_phase(copus)
    for i in copus:
        fout.write(i+" "+"nresume"+" "+str(10)+"\n")
    fout.close()
  #  fout.write()
def remove_n(text):
    intab = ""
    outtab = ""    
    deltab = "\n\t "
    trantab=text.maketrans(intab, outtab,deltab)
    return text.translate(trantab)

def generate_ngram(sentence, n=4, m=2):           # 生成n-gram
    if len(sentence) < n:
        n = len(sentence)
    temp=[tuple(sentence[i - k:i]) for k in range(m, n + 1) for i in range(k, len(sentence) + 1) ]                       # 生成2个或者3个的gram
    return [item for item in temp if len(''.join(item).strip())>1 and len(pattern.findall(''.join(item).strip()))==0]    # 去掉非字母汉字的符号
  
if __name__=="__main__":
    # 分字进行n-gram
    copus_character=[generate_ngram(line.strip())  for line  in open('text.txt','r',encoding='utf8') if len(line.strip())>0 and "RESUMEDOCSSTARTFLAG" not in line]    
    # 先用hanlp分词，在对词进行n-gram
    copus_word=[generate_ngram(seg_sentences(line.strip(),with_filter=True) ) for line  in open('text.txt','r',encoding='utf8') if len(line.strip())>0 and "RESUMEDOCSSTARTFLAG" not in line]
    copus_word=chain.from_iterable(copus_word)
    copus_word=['_'.join(item) for item in copus_word]
    fout=open("ngram2_3.txt", "w", encoding='utf-8')
    
    dic_filter={}                     # 统计词频
    for item in copus_word:
        if item in dic_filter:
            dic_filter[item]+=1
        else:
            dic_filter[item]=1
    sort_dic=dict(sorted(dic_filter.items(),key=lambda val:val[1],reverse=True))       #reverse=True为降序排列,返回list
    fout.write(json.dumps(sort_dic, ensure_ascii=False,cls=NumpyEncoder))               
    fout.close() 

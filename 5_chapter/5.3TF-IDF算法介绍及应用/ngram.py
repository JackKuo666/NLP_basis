#encoding=utf8
import json
import sys,os,re
import numpy
from tokenizer import seg_sentences
pattern=re.compile(u'[^a-zA-Z\u4E00-\u9FA5]')
from jpype import *
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer 
keep_pos="n,an,vn,nr,ns,nt,nz,nb,nba,nbc,nbp,nf,ng,nh,nhd,o,nz,nx,ntu,nts,nto,nth,ntch,ntcf,ntcb,ntc,nt,nsf,ns,nrj,nrf,nr2,nr1,nr,nnt,nnd,nn,nmc,nm,nl,nit,nis,nic,ni,nhm,nhd"
keep_pos_set=set(keep_pos.split(","))
stop_pos="q,b,f,p,qg,qt,qv,r,rg,Rg,rr,ry,rys,ryt,ryv,rz,rzs,rzt,rzv,s,v,vd,vshi,vyou,vf,vx,vl,vg,vf,vi,m,mq,uzhe,ule,uguo,ude1,ude2,ude3,usuo,udeng,uv,uzhe,uyy,udh,uls,uzhi,ulian,d,dl,u,c,cc,bl,ad,ag,al,a,r,q,p,z,pba,pbei,d,dl,o,e,xx,xu,y,yg,z,wkz,wky,wyz,wyy,wj,ww,wt,wd,wf,wm,ws,wp,wb,wh,wn,t,tg,vi,id,ip,url,tel"
stop_pos_set = set(stop_pos.split(','))
stop_ch='"是","由"'
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

def tokenize_raw(text):           # 先以标点符号为单位切分，再使用hanlp的seg_sentences分词
    split_sen=(i.strip() for i in re.split('。|,|，|：|:|？|！|\t|\n',_replace_c(text)) if len(i.strip())>5)    # 这里用()而不用[] 是因为（）是生成器，有利于减小内存，如果用[]生成list的话可能会内存不足
    return [seg_sentences(sentence) for sentence in split_sen]  




def list_2_ngram(sentence, n=4, m=2):         # n-gram
    if len(sentence) < n:
        n = len(sentence)
    temp=[tuple(sentence[i - k:i]) for k in range(m, n + 1) for i in range(k, len(sentence) + 1) ]
    return [item for item in temp if len(''.join(item).strip())>1 and len(pattern.findall(''.join(item).strip()))==0]

if __name__=="__main__":
    
    #'PlatformESB 组件 子系统 监控 NBI 接口'
    copus=[tokenize_raw(line.strip()) for line in open('text.txt','r',encoding='utf8') if len(line.strip())>0 and "RESUMEDOCSSTARTFLAG" not in line]
    #['TM_拓扑 拓扑_管理 TM_拓扑_管理']
    doc=[]
    if len(copus)>1: 
        for list_copus in copus:
            for t in list_copus:
                doc.extend([' '.join(['_'.join(i) for i in list_2_ngram(t, n=4, m=2)])])
    doc=list(filter(None,doc))                                   # 对分词进行n-gram，然后连接
    fout=open("ngram2_4.txt", "w", encoding='utf-8')

    # 使用tfidf计算频率
    vectorizer1=CountVectorizer()  #初始化一个计数类
    
    transformer=TfidfTransformer() #该类会统计每个词语的tf-idf权值
    freq1=vectorizer1.fit_transform(doc)  # 计算词频 65x1179
    tfidf=transformer.fit_transform(freq1)
    word_freq=[freq1.getcol(i).sum() for i in range(freq1.shape[1])]
                 
    tfidf_sum=[tfidf.getcol(i).sum() for i in range(tfidf.shape[1])]

    tfidf_dic=vectorizer1.vocabulary_
    tfidf_dic=dict(zip(tfidf_dic.values(),tfidf_dic.keys())) # 反转

    dic_filter={}
    def _add(wq,tf,i):
        dic_filter[tfidf_dic[i]]=[wq,tf]
    for i,(word_freq_one,w_one) in enumerate(zip(word_freq,tfidf_sum)):
        _add(word_freq_one, w_one, i)
    sort_dic=dict(sorted(dic_filter.items(),key=lambda val:val[1],reverse=True))#,reverse=True为降序排列,返回list
    fout.write(json.dumps(sort_dic, ensure_ascii=False,cls=NumpyEncoder))               
    fout.close() 
shutdownJVM()
                        #output_file.close() 
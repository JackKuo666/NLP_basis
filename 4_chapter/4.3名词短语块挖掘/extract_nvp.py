#encoding=utf8
import os,json,nltk,re
from jpype import *
from tokenizer import cut_hanlp
huanhang=set(['。','？','！','?'])
keep_pos="q,qg,qt,qv,s,t,tg,g,gb,gbc,gc,gg,gm,gp,mg,Mg,n,an,ude1,nr,ns,nt,nz,nb,nba,nbc,nbp,nf,ng,nh,nhd,o,nz,nx,ntu,nts,nto,nth,ntch,ntcf,ntcb,ntc,nt,nsf,ns,nrj,nrf,nr2,nr1,nr,nnt,nnd,nn,nmc,nm,nl,nit,nis,nic,ni,nhm,nhd"
keep_pos_nouns=set(keep_pos.split(","))
keep_pos_v="v,vd,vg,vf,vl,vshi,vyou,vx,vi,vn"
keep_pos_v=set(keep_pos_v.split(","))
keep_pos_p=set(['p','pbei','pba'])
merge_pos=keep_pos_p|keep_pos_v
keep_flag=set(['：','，','？','。','！','；','、','-','.','!',',',':',';','?','(',')','（','）','<','>','《','》'])
drop_pos_set=set(['xu','xx','y','yg','wh','wky','wkz','wp','ws','wyy','wyz','wb','u','ud','ude1','ude2','ude3','udeng','udh'])

def getNodes(parent,model_tagged_file):               # 使用for循环遍历树
    text=''
    for node in parent:
        if type(node) is nltk.Tree:                   # 如果是NP或者VP的合并分词
            if node.label() == 'NP':   
                text+=''.join(node_child[0].strip() for node_child in node.leaves())+"/NP"+3*" "
            if node.label() == 'VP':
                text+=''.join(node_child[0].strip() for node_child in node.leaves())+"/VP"+3*" "
        else:                                         # 不是树的，就是叶子节点，我们直接表解词PP或者其他O
            if node[1] in keep_pos_p:
                text+=node[0].strip()+"/PP"+3*" "  
            if node[0] in huanhang : 
                text+=node[0].strip()+"/O"+3*" "                    
            if node[1] not in merge_pos:
                text+=node[0].strip()+"/O"+3*" "                             
            #print("hh")
    model_tagged_file.write(text+"\n")     

def grammer(sentence,model_tagged_file):#{内/f 训/v 师/ng 单/b 柜/ng}
    """
    input sentences shape like :[('工作', 'vn'), ('描述', 'v'), ('：', 'w'), ('我', 'rr'), ('曾', 'd'), ('在', 'p')]
    """
                                # 定义名词块 “< >”:一个单元       “*”：匹配零次或多次    “+”：匹配一次或多次  “<ude1>?”： “的”出现零次或一次
    grammar1 = r"""NP:   
        {<m|mg|Mg|mq|q|qg|qt|qv|s|>*<a|an|ag>*<s|g|gb|gbc|gc|gg|gm|gp|n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|o|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+<f>?<ude1>?<g|gb|gbc|gc|gg|gm|gp|n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|o|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+}
        {<n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+<cc>+<n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+}
        {<m|mg|Mg|mq|q|qg|qt|qv|s|>*<q|qg|qt|qv>*<f|b>*<vi|v|vn|vg|vd>+<ude1>+<n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+}
        {<g|gb|gbc|gc|gg|gm|gp|n|an|nr|ns|nt|nz|nb|nba|nbc|nbp|nf|ng|nh|nhd|nz|nx|ntu|nts|nto|nth|ntch|ntcf|ntcb|ntc|nt|nsf|ns|nrj|nrf|nr2|nr1|nr|nnt|nnd|nn|nmc|nm|nl|nit|nis|nic|ni|nhm|nhd>+<vi>?}
        VP:{<v|vd|vg|vf|vl|vshi|vyou|vx|vi|vn>+}   
        """                                     # 动词短语块
    cp = nltk.RegexpParser(grammar1)
    try :
        result = cp.parse(sentence)             # nltk的依存语法分析，输出是以grammer设置的名词块为单位的树
    except:
        pass
    else:
        getNodes(result,model_tagged_file)      # 使用 getNodes 遍历树【这个是使用for循环，上一个是使用栈动态添加】


def data_read():
    fout=open('nvp.txt','w',encoding='utf8')
    for line in open('text.txt','r',encoding='utf8'):    
        line=line.strip()
        grammer(cut_hanlp(line),fout)     # 先进行hanlp进行分词，在使用grammer进行合并短语
    fout.close()

if __name__=='__main__':
    data_read()
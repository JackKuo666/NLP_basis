#encoding=utf8
import nltk,json

from .tools import ner_stanford,cut_stanford
keep_pos="q,qg,qt,qv,s,t,tg,g,gb,gbc,gc,gg,gm,gp,m,mg,Mg,mq,n,an,vn,ude1,nr,ns,nt,nz,nb,nba,nbc,nbp,nf,ng,nh,nhd,o,nz,nx,ntu,nts,nto,nth,ntch,ntcf,ntcb,ntc,nt,nsf,ns,nrj,nrf,nr2,nr1,nr,nnt,nnd,nn,nmc,nm,nl,nit,nis,nic,ni,nhm,nhd"
keep_pos_nouns=set(keep_pos.split(","))
keep_pos_v="v,vd,vg,vf,vl,vshi,vyou,vx,vi"
keep_pos_v=set(keep_pos_v.split(","))
keep_pos_p=set(['p','pbei','pba'])
def get_stanford_ner_nodes(parent):
    date=''
    org=''
    loc=''
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == 'DATE' :
                date=date+" "+''.join([i[0]  for i in node])
            elif node.label() == 'ORGANIZATIONL' :
                org=org+" "+''.join([i[0]  for i in node])
            elif node.label() == 'LOCATION': 
                loc=loc+" "+''.join([i[0]  for i in node])
    if len(date)>0 or len(org)>0  or len(loc)>0 :
        return {'date':date,'org':org,'loc':loc}    
    else:
        return {}
def grammer_parse(raw_sentence=None,file_object=None):
    #assert grammer_type in set(['hanlp_keep','stanford_ner_drop','stanford_pos_drop'])
    if len(raw_sentence.strip())<1:
        return False
    grammer_dict=\
    {
    
         'stanford_ner_drop': r"""
        DATE:{<DATE>+<MISC>?<DATE>*}
        {<DATE>+<MISC>?<DATE>*}
        {<DATE>+}
        {<TIME>+}
        ORGANIZATIONL:{<ORGANIZATION>+}
        LOCATION:{<LOCATION|STATE_OR_PROVINCE|CITY|COUNTRY>+}
        """     
    }
    
    stanford_ner_drop_rp = nltk.RegexpParser(grammer_dict['stanford_ner_drop'])
    try :
        stanford_ner_drop_result = stanford_ner_drop_rp.parse(ner_stanford(raw_sentence) )
        
    except:
        print("the error sentence is {}".format(raw_sentence))
    else:
        
        stanford_keep_drop_dict=get_stanford_ner_nodes(stanford_ner_drop_result)
        if len(stanford_keep_drop_dict)>0 :
            file_object.write(json.dumps(stanford_keep_drop_dict, skipkeys=False, 
                                        ensure_ascii=False, 
                                        check_circular=True, 
                                        allow_nan=True, 
                                        cls=None, 
                                        indent=4, 
                                        separators=None, 
                                        default=None, 
                                        sort_keys=False))

        
        
        

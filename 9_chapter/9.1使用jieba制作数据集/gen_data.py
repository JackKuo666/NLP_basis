#encoding=utf8
import os,jieba,csv
import jieba.posseg as pseg
c_root=os.getcwd()+os.sep+"source_data"+os.sep
dev=open("example.dev",'w',encoding='utf8')
train=open("example.train",'w',encoding='utf8')
test=open("example.test",'w',encoding='utf8')
biaoji = set(['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW','CL'])
fuhao=set(['。','?','？','!','！'])
dics=csv.reader(open("DICT_NOW.csv",'r',encoding='utf8'))
for row in dics:                                                   # 将医学专有名词以及标签加入结巴词典中
    if len(row)==2:
        jieba.add_word(row[0].strip(),tag=row[1].strip())            # add_word保证添加的词语不会被cut掉
        jieba.suggest_freq(row[0].strip())                           # 可调节单个词语的词频，使其能（或不能）被分出来。
split_num=0
for file in os.listdir(c_root):
    if "txtoriginal.txt" in file:
        fp=open(c_root+file,'r',encoding='utf8')
        for line in fp:
            split_num+=1
            words=pseg.cut(line)
            for key,value in words: 
                #print(key)
                #print(value)
                if value.strip() and key.strip():
                    import time 
                    start_time=time.time()
                    index=str(1) if split_num%15<2 else str(2)  if split_num%15>1 and split_num%15<4 else str(3) 
                    end_time=time.time()
                    #print("method one used time is {}".format(end_time-start_time))
                    if value not in biaoji:
                        value='O'
                        for achar in key.strip():
                            if achar and achar.strip() in fuhao:
                                string=achar+" "+value.strip()+"\n"+"\n"
                                dev.write(string) if index=='1' else test.write(string) if index=='2' else train.write(string) 
                            elif achar.strip() and achar.strip() not in fuhao:
                                string = achar + " " + value.strip() + "\n"
                                dev.write(string) if index=='1' else test.write(string) if index=='2' else train.write(string) 
        
                    elif value.strip()  in biaoji:
                        begin=0
                        for char in key.strip():
                            if begin==0:
                                begin+=1
                                string1=char+' '+'B-'+value.strip()+'\n'
                                if index=='1':                               
                                    dev.write(string1)
                                elif index=='2':
                                    test.write(string1)
                                elif index=='3':
                                    train.write(string1)
                                else:
                                    pass
                            else:
                                string1 = char + ' ' + 'I-' + value.strip() + '\n'
                                if index=='1':                               
                                    dev.write(string1)
                                elif index=='2':
                                    test.write(string1)
                                elif index=='3':
                                    train.write(string1)
                                else:
                                    pass
                    else:
                        continue                        
dev.close()
train.close()
test.close()            
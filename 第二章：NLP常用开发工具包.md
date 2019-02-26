# 第二章:NLP常用开发工具包
# 1.NumPy
numpy系统是Python的一种开源的数值计算包。 包括：1、一个强大的N维数组对象Array；2、比较成熟的（广播）函数库；3、用于整合C/C++和Fortran代码的工具包；4、实用的线性代数、傅里叶变换和随机数生成函数。numpy和稀疏矩阵运算包scipy配合使用更加方便。
```
conda install numpy
```

# 2. NLTK       
Natural Language Toolkit，自然语言处理工具包，在NLP领域中， 最常使用的一个Python库。 
```
conda install nltk
```
# 3.Gensim
Gensim是一个占内存低，接口简单，免费的Python库，它可以用来从文档中自动提取语义主题。它包含了很多非监督学习算法如：TF/IDF，潜在语义分析（Latent Semantic Analysis，LSA）、隐含狄利克雷分配（Latent Dirichlet Allocation，LDA），层次狄利克雷过程 （Hierarchical Dirichlet Processes，HDP）等。
Gensim支持Word2Vec,Doc2Vec等模型。 
```
conda install gensim
```
# 4.Tensorflow 
TensorFlow是谷歌基于DistBelief进行研发的第二代人工智能学习系统。TensorFlow可被用于语音识别或图像识别等多项机器学习和深度学习领域。TensorFlow是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。它灵活的架构让你可以在多种平台上展开计算，例如台式计算机中的一个或多个CPU（戒GPU），服务器，移动设备等等。TensorFlow 最初由Google大脑小组（隶属于Google机器智能研究机构）的研究员和工程师们开发出来，用于机器学习和深度神经网络方面的研究，但这个系统的通用 性使其也可广泛用于其他计算领域。 
```
conda install tensorflow
```
# 5.jieba 
“结巴”中文分词：是广泛使用的中文分词工具，具有以下特点： 
1）三种分词模式：精确模式，全模式和搜索引擎模式 
2）词性标注和返回词语在原文的起止位置（ Tokenize） 
3）可加入自定义字典 
4）代码对 Python 2/3 均兼容 
5）支持多种语言，支持简体繁体 
项目地址：https://github.com/fxsjy/jieba 
```
pip install jieba
```
# 6.Stanford NLP
Stanford NLP提供了一系列自然语言分析工具。它能够给出基本的 词形，词性，不管是公司名还是人名等，格式化的日期，时间，量词， 并且能够标记句子的结构，语法形式和字词依赖，指明那些名字指向同 样的实体，指明情绪，提取发言中的开放关系等。  1.一个集成的语言分析工具集； 2.进行快速，可靠的任意文本分析； 3.整体的高质量的文本分析; 4.支持多种主流语言; 5.多种编程语言的易用接口; 6.方便的简单的部署web服务。 
## 安裝
```
Python 版本stanford nlp 安装
• 1)安装stanford nlp自然语言处理包: pip install stanfordcorenlp
• 2)下载Stanford CoreNLP文件
https://stanfordnlp.github.io/CoreNLP/download.html
• 3)下载中文模型jar包, http://nlp.stanford.edu/software/stanford-chinese-
corenlp-2018-02-27-models.jar,
• 4)把下载的stanford-chinese-corenlp-2018-02-27-models.jar
放在解压后的Stanford CoreNLP文件夹中，改Stanford CoreNLP文件夹名为stanfordnlp（可选）
• 5)在Python中引用模型:
• from stanfordcorenlp import StanfordCoreNLP
• nlp = StanfordCoreNLP(r‘path', lang='zh')
例如：
nlp = StanfordCoreNLP(r'/home/kuo/NLP/module/stanfordnlp/', lang='zh')
```
## 测试
```
#-*-encoding=utf8-*-
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'/home/kuo/NLP/module/stanfordnlp/', lang='zh')

fin=open('news.txt','r',encoding='utf8')
fner=open('ner.txt','w',encoding='utf8')
ftag=open('pos_tag.txt','w',encoding='utf8')
for line in fin:
    line=line.strip()
    if len(line)<1:
        continue
 
    fner.write(" ".join([each[0]+"/"+each[1] for  each in nlp.ner(line) if len(each)==2 ])+"\n")
    ftag.write(" ".join([each[0]+"/"+each[1] for each in nlp.pos_tag(line) if len(each)==2 ])+"\n")
fner.close()   
ftag.close()
print ("okkkkk")
```
# 7.Hanlp
HanLP是由一系列模型与算法组成的Java工具包，目标是普及自然 语言处理在生产环境中的应用。HanLP具备功能完善、性能高效、架构 清晰、语料时新、可自定义的特点。       功能：中文分词 词性标注 命名实体识别 依存句法分析 关键词提取 新词发现 短语提取 自动摘要 文本分类 拼音简繁 

## Hanlp环境安装
```
• 1、安装Java:我装的是Java 1.8
• 2、安裝Jpype,
> conda install -c conda-forge jpype1
>[或者]pip install jpype1
• 3、测试是否按照成功:
from jpype import *
startJVM(getDefaultJVMPath(), "-ea")
java.lang.System.out.println("Hello World")
shutdownJVM()       
```
## Hanlp安装
```
• 1、下载hanlp.jar包: https://github.com/hankcs/HanLP
• 2、下载data.zip: https://github.com/hankcs/HanLP/releases 中
http://hanlp.linrunsoft.com/release/data-for-1.7.0.zip 后解压数据包。
• 3、配置文件
• 示例配置文件:hanlp.properties
• 配置文件的作用是告诉HanLP数据包的位置,只需修改第一行:root=usr/home/HanLP/
• 比如data目录是/Users/hankcs/Documents/data,那么root=/Users/hankcs/Documents/

```
## 测试
```
#-*- coding:utf-8 -*-
from jpype import *

startJVM(getDefaultJVMPath(), "-Djava.class.path=/home/kuo/NLP/module/hanlp/hanlp-1.6.2.jar:/home/kuo/NLP/module/hanlp",
         "-Xms1g",
         "-Xmx1g") # 启动JVM，Linux需替换分号;为冒号:

print("=" * 30 + "HanLP分词" + "=" * 30)
HanLP = JClass('com.hankcs.hanlp.HanLP')
# 中文分词
print(HanLP.segment('你好，欢迎在Python中调用HanLP的API'))
print("-" * 70)
```

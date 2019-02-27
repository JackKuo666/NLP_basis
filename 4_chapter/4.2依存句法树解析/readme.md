# 说明

## 使用Stanfordnlp和nltk进行**依存句法分析**,提取动名词短语

分词之后名词动词合并成chunking短语


主函数：sentenceSplit_host.py

输入：text.txt

输出：dependency.txt

## 主要步骤
通过读取text.txt文本,

进行Stanfordnlp的parse进行**依存句法分析**，

然后将分析结果通过nltk构建成树结构，

通过nltk构建的search方法进行**子树搜索，递归遍历搜索，叶子节点提取**返回需要的（动词，名词短语）对

注意：这个遍历树的方法是使用栈【动态递归】
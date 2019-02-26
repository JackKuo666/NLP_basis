# 说明
这个项目含有两个脚本，三个文本：
```
cut_data.py     主函数
tokenizer.py    hanlp函数
dict.txt        jieba分词的自定义词典
text.txt        原始文本
result_cut.txt  分词后的输出文本
```
1.主函数按行读取text.txt文本，分别用jieba分词和hanlp分词进行分词，

2.1.添加jieba分词自定义词典,在./dict.txt中添加：
```
台中
氟尿嘧啶单药
联合奥沙利铂
Cox模型
TNM分期
```
2.2.添加hanlp分词自定义词典，在"/home/kuo/NLP/module/hanlp/data/dictionary/custom/"下

2.2.1.删除"CustomDictionary.txt.bin"

2.2.2.在“CustomDictionary.txt”中添加
```
数据库设计 n 4729
TNM分期 n 4729
工作描述 n 4729
```
3.利用正则匹配去弥补字典也解决不了的特殊字符


# 运行
```
python cut_data.py
```
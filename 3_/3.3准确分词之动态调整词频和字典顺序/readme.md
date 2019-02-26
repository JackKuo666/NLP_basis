# 说明
准确分词之动态调整词频和字典顺序
## 1.jieba动态调整词频
四中方法：
### 1.1
```python
#jieba.load_userdict("dict.txt")
```
### 1.2.
```python
# # 设置高词频：一个
# jieba.suggest_freq('台中',tune=True)
```
### 1.3.
```python
# 设置高词频：dict.txt中的每一行都设置一下
# fp=open("dict.txt",'r',encoding='utf8')
# for line in fp:
#     line=line.strip()
#     jieba.suggest_freq(line, tune=True)
```
### 1.4.
```python
# # 设置高词频：dict.txt中的每一行都设置一下快速方法
# [jieba.suggest_freq(line.strip(), tune=True) for line in open("dict.txt",'r',encoding='utf8')]
```

## 2.hanlp动态调整字典顺序
```
python sort_dict_by_lenth.py 
```
该脚本目的是按照每行词的数量重新排序，数量多的排前面，使得切词的时候尽量完整切词
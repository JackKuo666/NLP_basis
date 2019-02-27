# 说明

利用hanlp进行分词，然后使用n-gram进行拼接，然后使用计算tfidf，按照词频排序。

# 其他
这里用()而不用[] 是因为()是生成器，有利于减小内存，如果用[]生成list的话可能会内存不足
```python
split_sen=(i.strip() for i in re.split('。|,|，|：|:|？|！|\t|\n',_replace_c(text)) if len(i.strip())>5)    
```

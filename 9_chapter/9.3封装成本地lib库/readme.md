# 说明
将测试环节封装成一个类Chunking，然后一次调用，一次初始化就可以处理很多数据。

主要内容在

get_text_input() 类Chunk的方法 

evaluate_line()  类Model的方法

## 封装成lib的意思是，我们如果想要使用，直接初始化Chunk类，并调用方法就行了，例子如下：

```python
    c=Chunk()                                              # 初始化chunk类，在chunk类初始化的时候调用Model类，同时初始化Model类
    for line in open('text.txt','r',encoding='utf8'):
        print(c.get_text_input(line.strip()))
 
```
# 说明
主函数：
extract_data.py

## hanlp去停用词
### 1.在停用词表找你想忽略的停用词符号
```python
drop_pos_set=set(['xu','xx','y','yg','wh','wky','wkz','wp','ws','wyy','wyz','wb','u','ud','ude1','ude2','ude3','udeng','udh','p','rr','w'])

```
例如：想去掉“标点符号”，就在上边的代码中增加“w”
### 2.利用hanlp分词，然后去掉上边所示的词性的词
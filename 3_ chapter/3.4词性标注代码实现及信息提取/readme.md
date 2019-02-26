# 说明
## hanlp去停用词
### 1.在停用词表找你想忽略的停用词符号
### 2.在如下代码中更改就行了
```python
drop_pos_set=set(['xu','xx','y','yg','wh','wky','wkz','wp','ws','wyy','wyz','wb','u','ud','ude1','ude2','ude3','udeng','udh','p','rr','w'])

```
例如：想去掉“标点符号”，就在上边的代码中增加“w”
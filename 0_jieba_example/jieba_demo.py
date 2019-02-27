# encoding=utf-8
import jieba
import jieba.posseg as pseg

print("\njieba分词全模式：")
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

print("\njieba分词精确模式：")
seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

print("\njieba默认分词是精确模式：")
seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

print("\njiba搜索引擎模式：")
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))

strings="是广泛使用的中文分词工具，具有以下特点："
words = pseg.cut(strings)

print("\njieba词性标注：")
for word, flag in words:
    print('%s %s' % (word, flag))
#encoding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
filename = 'text8.zip'

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)


# step 1剔除高频停用词减少模型噪音，并加速训练
def remove_fre_stop_word(words):
    t = 1e-5  # t 值
    threshold = 0.8  # 剔除概率阈值

    # 统计单词频率
    int_word_counts = collections.Counter(words)
    total_count = len(words)
    # 计算单词频率
    word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
    # 计算被删除的概率
    prob_drop = {w: 1 - np.sqrt(t / f) for w, f in word_freqs.items()}
    # 对单词进行采样
    train_words = [w for w in words if prob_drop[w] < threshold]

    return train_words

words = remove_fre_stop_word(words)



# Step 2: Build the dictionary and replace rare words with UNK token.
# vocabulary_size = len(words)
vocabulary_size = len(set(words)) # words 中不重复的分词数量
print('Data size', vocabulary_size)
def build_dataset(words):
    count = [['UNK', -1]]
    #collections.Counter(words).most_common
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))  # words中每个分词计数，然后按照词频降序排列放在count里：[['UNK', -1], ('的', 99229), ('在', 25925), ('是', 20172), ('年', 17007), ('和', 16514), ('为', 15231), ('了', 13053), ('有', 11253), ('与', 11194)]
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)                                     # count中每个词分配一个编号，：[('UNK', 0), ('的', 1), ('在', 2), ('是', 3), ('年', 4), ('和', 5), ('为', 6), ('了', 7), ('有', 8), ('与', 9)]
                                                                               # 相当于词典，key是分词，value是分配的编号
    data = list()
    unk_count = 0

    data=[dictionary[word]  if  word in dictionary else 0 for word in words]   # 将words中的每个分词用序列号表示:[14880, 4491, 483, 70, 1, 1009, 1850, 317, 14, 76]

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))     # 将dictionary中的key和value对换:[(0, 'UNK'), (1, '的'), (2, '在'), (3, '是'), (4, '年'), (5, '和'), (6, '为'), (7, '了'), (8, '有'), (9, '与')]
                                                                               # 相当于key是编号，value是对应的词
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)             # data：2262896,语料中的每个词的对应的编号； count:199247，相当于词频表，key是语料中所有的词，value是词频；
                                                                               # dictionary：199247，这个语料对应的词典，key是词，value是唯一编号； reverse_dictionary：199247，这个语料对应的词典，key是唯一编号，value是词；
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0





# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index                                                           # 使用全局变量，意思是在函数里边也能更改其值
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)                           # 类似于list
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):#i取值0,1,2
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])                        # buffer队列，先进先出，永远保持5个
        data_index = (data_index + 1) % len(data)
    data_index -= 1                       # 这里修复一个bug，原本['欧几里得', '西元前', '三', '希腊', '数学家', '几何', '父', '此画', '拉斐尔', '雅典','数量']
                                          # 输入按顺序应该是：batch1：三，希腊，batch2：拉斐尔，雅典，但是这里data_index 在最后一次循环多加1，导致batch2：雅典，数量
                                          # 所以这里要减去1
    return batch, labels

for j in range(10):
    batch, labels = generate_batch(batch_size=8, num_skips=4, skip_window=2)    # skip_window=2代表着选取左input word左侧2个词和右侧2个词进入我们的窗口，所以整个窗口大小span=2x2=4
                                                                                # num_skips，它代表着我们从整个窗口中选取多少个不同的词作为我们的output word
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]],
            '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

# hyperparameters
batch_size = 128
embedding_size = 128 # dimension of the embedding vector
skip_window = 2 # how many words to consider to left and right
num_skips = 4 # how many times to reuse an input to generate a label

# we choose random validation dataset to sample nearest neighbors
# here, we limit the validation samples to the words that have a low
# numeric ID, which are also the most frequently occurring words
valid_size = 16 # size of random set of words to evaluate similarity on
valid_window = 100 # only pick development samples from the first 'valid_window' words
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64 # number of negative examples to sample

# create computation graph
graph = tf.Graph()

with graph.as_default():
    # input data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # operations and variables
    # look up embeddings for inputs
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
    # loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
    #                  labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))

    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(nce_weights, nce_biases, train_labels,
                                   embed, num_sampled, vocabulary_size))

    # 这里设置num_sampled=num_sampled就是在负采样的时候默认执行 P(k) = (log(k + 2) - log(k + 1)) / log(range_max + 1)
    
    # construct the SGD optimizer using a learning rate of 1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # compute the cosine similarity between minibatch examples and all embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # add variable initializer

#5
num_steps = 100001

with tf.Session(graph=graph) as session:
    # we must initialize all variables before using them
    tf.initialize_all_variables().run()
    print('initialized.')
    
    # loop through all training steps and keep track of loss
    average_loss = 0
  
    for step in xrange(num_steps):
        # generate a minibatch of training data
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        
        # we perform a single update step by evaluating the optimizer operation (including it
        # in the list of returned values of session.run())
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val


        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # the average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
        
        # computing cosine similarity (expensive!)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                # get a single validation sample
                valid_word = reverse_dictionary[valid_examples[i]]
                # number of nearest neighbors
                top_k = 8
                # computing nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary.get(nearest[k],None)
                    #close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
        
    final_embeddings = normalized_embeddings.eval()
    print("*"*10+"final_embeddings:"+"*"*10+"\n",final_embeddings)
    fp=open('vector_skip_gram.txt','w',encoding='utf8')
    for k,v in reverse_dictionary.items():
        t=tuple(final_embeddings[k])

        s=''
        for i in t:
            i=str(i)
            s+=i+" "
            
        fp.write(v+" "+s+"\n")

    fp.close()



# Step 6: Visualize the embeddings.
import matplotlib.pyplot as plt
def plot_with_labels(low_dim_embs, plot_labels, filename='tsne_skip_gram.png'):
    assert low_dim_embs.shape[0] >= len(plot_labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(plot_labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(u'{}'.format(label),
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

    plt.savefig(filename)




try:
    from sklearn.manifold import TSNE
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文字符
    mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示正负号

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    plot_labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, plot_labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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



# Step 3: Function to generate a training batch for the skip-gram model.
data_index = 0
def generate_batch(batch_size, bag_window):
    global data_index
    span = 2 * bag_window + 1 # [ bag_window target bag_window ]
    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size):
        # just for testing
        buffer_list = list(buffer)
        labels[i, 0] = buffer_list.pop(bag_window)
        batch[i] = buffer_list
        # iterate to the next buffer
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:16]])

for bag_window in [1, 2]:
    data_index = 0
    batch, labels = generate_batch(batch_size=4, bag_window=bag_window)
    print('\nwith bag_window = %d:' % (bag_window))
    print('    batch:', [[reverse_dictionary[w] for w in bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(4)])

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
bag_window = 2  # How many words to consider left and right.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size, bag_window * 2])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embeds = tf.nn.embedding_lookup(embeddings, train_dataset)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, train_labels,
                                   tf.reduce_sum(embeds, 1), num_sampled, vocabulary_size))

    # Optimizer.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps = 100001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(
            batch_size, bag_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()

    print("*" * 10 + "final_embeddings:" + "*" * 10 + "\n", final_embeddings)
    fp = open('vector_cbow.txt', 'w', encoding='utf8')
    for k, v in reverse_dictionary.items():
        t = tuple(final_embeddings[k])

        s = ''
        for i in t:
            i = str(i)
            s += i + " "

        fp.write(v + " " + s + "\n")

    fp.close()


# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, plot_labels, filename='tsne_cbow.png'):
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

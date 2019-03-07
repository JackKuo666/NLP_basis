#-*-encoding=utf8-*-
import json
import os
import sys
from itertools import chain
import tensorflow as tf
import numpy as np
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
currentPath=os.getcwd()
sys.path.append(currentPath)
import jieba
root_path=os.getcwd()
global pyversion
if sys.version>'3':
    pyversion='three'
else:
    pyversion='two'
if pyversion=='three':
    import pickle
else :
    import cPickle,pickle
root_path=os.getcwd()+os.sep

CONFIG= {
   
}

class Model(object):
    #初始化模型参数
    def __init__(self, config):

        self.config = config
        
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]#样本中总字数
        self.num_segs = 4
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        self.model_type = config['model_type']
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3
        self.num_filter = self.lstm_dim 
        self.embedding_dim = self.char_dim + self.seg_dim
        self.repeat_times = 4
        self.cnn_output_width = 0
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)          # 调用embedding_layer
        if self.model_type == 'idcnn':
            model_inputs = tf.nn.dropout(embedding, self.dropout)
            model_outputs = self.IDCNN_layer(model_inputs)                                   # 调用IDCNN_layer
            self.logits = self.project_layer_idcnn(model_outputs)                            # 调用project_layer_idcnn
        
        else:
            raise KeyError

        self.loss = self.loss_layer(self.logits, self.lengths)                               # 调用loss_layer

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):

        embedding = []
        self.char_inputs_test=char_inputs
        self.seg_inputs_test=seg_inputs
        with tf.variable_scope("char_embedding" if not name else name):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        #shape=[4*20]
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        self.embed_test=embed
        self.embedding_test=embedding
        return embed

    def IDCNN_layer(self, model_inputs, name=None):

        model_inputs = tf.expand_dims(model_inputs, 1)
        self.model_inputs_test=model_inputs
        reuse = False
        if self.dropout == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            #shape=[1*3*120*100]
            shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter]
            print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer",use_cudnn_on_gpu=True)
            self.layerInput_test=layerInput
            finalOutFromLayers = []
            
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    #1,1,2
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        if j==1 and i==1:
                            self.w_test_1=w
                        if j==2 and i==1:
                            self.w_test_2=w                            
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        self.conv_test=conv 
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)
            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_idcnn(self, idcnn_outputs, name=None):

        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name='crf_loss'):  
        with tf.variable_scope(name):  
            small = -1000.0
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)              
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

# ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
#                       evaluate_line（）函数
# ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××

    def create_feed_dict(self, batch):
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        return feed_dict

    def run_step(self, sess, batch):
        feed_dict = self.create_feed_dict(batch)
        lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
        return lengths, logits

    def decode(self, logits, lengths, matrix):
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def result_to_json(self,string, tags):
        item = {"string": string, "entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        for char, tag in zip(string, tags):
            if tag[0] == "S":
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "E":
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
                entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
        return item

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval(session=sess)                       # crf 输出     ？待定，明天看训练网络
        lengths, scores = self.run_step(sess, inputs)               # 输入特征，输出每个字的分数：也就是logits
        batch_paths = self.decode(scores, lengths, trans)           # 将分数转化为label编号
        tags = [id_to_tag[idx] for idx in batch_paths[0]]           # 将label编号转化为标签
        return self.result_to_json(inputs[0][0], tags)

class Chunk(object):
    def __init__(self): 
        self.config_file=json.load(open("config_file", encoding="utf8"))
        self.tf_config = tf.ConfigProto()
        self.sess=tf.Session(config=self.tf_config)
        self.sess.run(tf.global_variables_initializer())         # 初始化所有全局变量
        self.maps="maps.pkl"
        if pyversion=='three':    
            self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = pickle.load(open(self.maps, "rb"))
        else:
            self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = pickle.load(open(self.maps, "rb"),protocol=2)   

        self.model = Model(self.config_file)                     # 初始化Model类

        self.ckpt = tf.train.get_checkpoint_state("ckpt")        # 读取模型参数
        if self.ckpt and tf.train.checkpoint_exists(self.ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % self.ckpt.model_checkpoint_path)
            self.model.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        else:
            print("No model file")
        
    def features(self,string):   # 分词之后的词的词嵌入特征：分词长为1的，特征为0；分词长度大于1的，特征为1,2...2，3
        def _w2f(word):
            lenth=len(word)
            if lenth==1:
                r=[0]
            if lenth>1:
                r=[2]*lenth
                r[0]=1
                r[-1]=3
            return r
        return list(chain.from_iterable([_w2f(word) for word in jieba.cut(string) if len(word.strip())>0]))
      
        
    def get_text_input(self,text):
        inputs = list()
        inputs.append([text])
        D = self.char_to_id["<UNK>"]
         
        inputs.append([[self.char_to_id.setdefault(char, D) 
                            for char in text if len(char.strip())>0]])
        inputs.append([self.features(text)])             # 分词之后，词嵌入的特征：长度是1的特征为0；分词长度大于1的，特征为1,2...2，3
        inputs.append([[]])        
        if len(text.strip())>1: 
            return self.model.evaluate_line(self.sess,inputs, self.id_to_tag) # inputs有三个输入特征：inputs[0]是原文；inputs[1]是字的序号；inputs[2]是分词之后词长的特征

        
if __name__ == "__main__":   
    c=Chunk()                                              # 初始化chunk类，在chunk类初始化的时候调用Model类，同时初始化Model类
    for line in open('text.txt','r',encoding='utf8'):
        print(c.get_text_input(line.strip()))
    #s="典型胸痛 因体力活动、情绪激动等诱发，突感心前区疼痛，多为发作性绞痛或压榨痛，也可为憋闷感。疼痛从胸骨后或心前区开始，向上放射至左肩、臂，甚至小指和无名指，休息或含服硝酸甘油可缓解。胸痛放散的部位也可涉及颈部、下颌、牙齿、腹部等。胸痛也可出现在安静状态下或夜间，由冠脉痉挛所致，也称变异型心绞痛。如胸痛性质发生变化，如新近出现的进行性胸痛，痛阈逐步下降，以至稍事体力活动或情绪激动甚至休息或熟睡时亦可发作。疼痛逐渐加剧、变频，持续时间延长，祛除诱因或含服硝酸甘油不能缓解，此时往往怀疑不稳定心绞痛。"
    #print(c.get_text_input(s))
  


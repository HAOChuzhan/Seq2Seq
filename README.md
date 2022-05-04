# Seq2Seq
基于Pytorch和Tensorflow两个版本的Seq2Seq模型，集成Attention、BeamSearch、Dropout等优化方法。

包含有闲聊问答 (seq2seq_dialog)、检索问答 (seq2seq_dialog)、任务型问答 (task_dialog)、主题问答 (theme_dialog)等问答形式。

### 文件说明

#### **1.config.py参数配置文件**

主要进行模型超参数以及相关文件路径的配置

#### **2.DataProcessing.py 预处理文件**

主要进行语料库的处理工作，包括语料处理、编码索引、生成语料库的词向量文件emb等。

#### **3.read_vecor.py 修改词向量文件**

原始词向量是由维基百科语料word2vec训练得到的，现在要对原始词向量进行一定的修改，

#### **4.SequenceToSequence.py Seq2Seq模型**

#### **5.Train.py 训练文件**

运算只需要运行此文件即可

#### **6.RestfulAPI.py**

运行此文件，然后打开index.html，即可进行人机对话。

#### **7.相关数据文件来源**

| 文件名称                          | 解释                                                     |
| --------------------------------- | -------------------------------------------------------- |
| clean_chat_corpus/xiaohuangji.tsv | 小黄鸡训练语料                                           |
| model/                            | 训练好的模型文件，可直接加载                             |
| data/data.pkl                     | 原始语料预处理之后的数据                                 |
| data/wiki.zh.text.vector          | 原始词向量文件                                           |
| data/word_vec.pkl                 | 修改后的词向量文件                                       |
| data/emb.pkl                      | 根据语料库的词语抽取出的词向量文件，用于embedding_lookup |
| data/w2i.pkl                      | 词与索引对应的文件                                       |

下载链接：[百度网盘](https://pan.baidu.com/s/1X2fixauTOE7RBkojBD90Pw)  提取码：yvxd 
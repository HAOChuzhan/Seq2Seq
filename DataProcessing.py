# encoding: utf-8

"""
    数据处理单元
    处理原始语料数据
    生成批训练数据
"""

import re
import os
import pdb
import pickle # Pickle模块将任意一个Python对象转换成一系统字节的这个操作过程叫做串行化对象。
import collections
import jieba
import codecs
import itertools
from gensim import models 
import random
from search_dialog.search_config import vocab_path
from CONFIG import data_config
import numpy as np

class DataUnit(object):

    # 特殊标签
    PAD = '<PAD>'  # 填充
    UNK = '<UNK>'  # 未知
    START = '<SOS>'
    END = '<EOS>'

    # 特殊标签的索引
    START_INDEX = 0
    END_INDEX =1
    UNK_INDEX = 2
    PAD_INDEX = 3

    def __init__(self, path, processed_path,
                 min_q_len, max_q_len,
                 min_a_len, max_a_len,
                 word2index_path, emb_path,
                 word_vec_path):
        """
            初始化函数，参数意义可查看CONFIG.py文件中的注释
        :param
        """
        self.path = path
        self.processed_path = processed_path
        self.word2index_path = word2index_path
        self.word_vec_path = word_vec_path
        self.emb_path = emb_path
        self.min_q_len = min_q_len
        self.max_q_len = max_q_len
        self.min_a_len = min_a_len
        self.max_a_len = max_a_len
        self.vocab_size = 0
        self.index2word = {}
        self.word2index = {}
        self.data = self.load_data()  # 加载处理好的语料
        self._fit_data_()       # 得到处理后语料库的所有词，并将其编码为索引值
        self.emb_write()

    def next_batch(self, batch_size):
        """
        生成一批训练数据
        :param batch_size: 每一批数据的样本数
        :return: 经过了填充处理的QA对
        """
        data_batch = random.sample(self.data, batch_size)
        batch = []
        for qa in data_batch:
            encoded_q = self.transform_sentence(qa[0])  # 将句子转化为索引，qa[0]代表question
            encoded_a = self.transform_sentence(qa[1])  # qa[1]代表answer


            # 填充句子
            q_len = len(encoded_q)
            encoded_q = encoded_q + [self.func_word2index(self.PAD)] * (self.max_q_len - q_len)

            encoded_a = encoded_a + [self.func_word2index(self.END)]
            a_len = len(encoded_a)
            encoded_a = encoded_a + [self.func_word2index(self.PAD)] * (self.max_a_len + 1 - a_len)

            batch.append((encoded_q, q_len, encoded_a, a_len))
        # print(batch)
        '''
        [([2330, 12211, 320, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3, [4229, 4229, 7616, 9383, 3968, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6), ([11674, 7583, 7760, 5824, 8683, 3913, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6, [11674, 8539, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([2784, 7541, 7760, 8683, 1226, 2366, 1133, 10025, 9383, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 9, [4229, 4229, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([1505, 9383, 10255, 5634, 3738, 5494, 3611, 11171, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 8, [4920, 9383, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([1121, 568, 1226, 4102, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4, [4920, 9383, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([12033, 1898, 1054, 8541, 12423, 11364, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6, [1388, 704, 4107, 3892, 11204, 533, 1226, 5458, 10131, 1536, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 11), ([4887, 208, 25, 1749, 6137, 1133, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6, [12211, 7616, 1708, 12211, 7616, 25, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 8), ([4437, 3738, 7238, 3913, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4, [5659, 6262, 10301, 548, 6124, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6), ([4110, 4920, 3968, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3, [10025, 9383, 7692, 209, 4107, 1388, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 7), ([753, 3405, 7881, 2045, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4, [7616, 9383, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([11668, 753, 7787, 4242, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4, [4394, 9478, 4107, 6380, 10878, 9484, 5198, 3968, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        3, 3], 9), ([12269, 2817, 3918, 4360, 6380, 5487, 12211, 12278, 4920, 4426, 3049, 1226, 3, 3, 3, 3, 3, 3, 3, 3], 12, [5487, 7616, 320, 9383, 5471, 12256, 631, 320, 12033, 10728, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 11), ([4229, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, [4920, 9383, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([5494, 3496, 1077, 4101, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        3, 3, 3, 3, 3, 3], 4, [1607, 3524, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([2697, 1898, 7960, 8747, 5227, 6367, 10499, 4711, 3447, 8465, 9134, 4920, 3, 3, 3, 3, 3, 3, 3, 3], 12, [7583, 3485, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([4576, 1133, 1897, 8937, 812, 4179, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6, [1406, 9656, 631, 12609, 12333, 4107, 6380, 5637, 9656, 9383, 3968, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3], 12), ([5299, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, [4920, 9383, 3968, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([12914, 3918, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2, [4920, 9383, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([1905, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, [6264, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2), ([11674, 12441, 7616, 6112, 2591, 9383, 6367, 1691, 1897, 9576, 3968, 3, 3, 3, 3, 3, 3, 3, 3, 3], 11, [13, 7405, 9383, 888, 6006, 12578, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 7), ([6498, 5265, 320, 4013, 5265, 472, 10333, 197, 11606, 3949, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 10, [9965, 5471, 3717, 25, 8239, 7923, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 8), ([4426, 631, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2, [4426, 9383, 3968, 7760, 1589, 3738, 5668, 6262, 9383, 3968, 4426, 9383, 1388, 1, 3, 3, 3, 3, 3, 3, 3], 14), ([5299, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, [4920, 
        9383, 12736, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([4107, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, [2915, 3878, 3953, 6367, 264, 6367, 3426, 11433, 11204, 11085, 548, 3968, 1, 3, 3, 3, 3, 3, 3, 3, 3], 13), ([7616, 9383, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2, [4920, 9383, 3968, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([1897, 5133, 1607, 10353, 6650, 8630, 11353, 3913, 6367, 5612, 3649, 1226, 3, 3, 3, 3, 3, 3, 3, 3], 12, [1406, 6124, 5612, 11353, 1226, 4107, 8465, 2891, 3497, 3968, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 11), ([1589, 7548, 11833, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3, [9835, 7215, 11813, 1781, 10131, 6262, 11833, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 9), ([1133, 907, 4048, 7760, 1771, 3913, 3738, 9459, 197, 6124, 4107, 3, 3, 3, 3, 3, 3, 3, 3, 3], 11, [5612, 7616, 1772, 1226, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6), ([7616, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, [4920, 9383, 3968, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([4920, 1226, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2, [4920, 9383, 5612, 516, 7011, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6), ([7616, 1610, 9383, 11213, 3918, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 5, [7616, 9383, 3968, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([6368, 5147, 1226, 11129, 5494, 3611, 4107, 11171, 4194, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 9, [7582, 3485, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([5598, 1653, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2, [4920, 9383, 3968, 1607, 3485, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6), ([4920, 9383, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2, [4229, 3956, 5211, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([8465, 12441, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2, [4920, 9383, 7583, 10131, 6262, 3510, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 7), ([11142, 7616, 5637, 9416, 9383, 3913, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6, [5653, 9383, 1388, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([11835, 11291, 6137, 6997, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4, [6772, 548, 7071, 10982, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 5), ([1905, 11668, 8251, 12526, 8392, 8539, 2773, 8798, 7616, 3913, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 10, [3701, 8797, 1226, 
        9588, 9383, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6), ([6772, 1226, 1607, 3611, 3913, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        3], 5, [7616, 9383, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([7818, 5270, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2, [1607, 5284, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([6367, 2708, 1133, 1226, 973, 10552, 12326, 801, 5659, 6367, 7286, 10555, 152, 3, 3, 3, 3, 3, 3, 3], 13, [10555, 10890, 219, 9383, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6), ([12033, 7818, 9647, 3738, 11849, 3949, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6, [5211, 9151, 7761, 8683, 3444, 5458, 7760, 9645, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 10), ([1898, 3738, 2, 1226, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4, [2, 9383, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([6124, 10878, 7310, 1226, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4, [5663, 2073, 2073, 5129, 5663, 2073, 2083, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 8), ([1898, 5230, 11674, 9206, 7616, 7415, 9383, 3918, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 8, [7511, 12211, 7616, 11615, 9383, 4107, 1962, 288, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 9), ([6137, 2466, 57, 2103, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4, [1426, 5830, 12788, 12723, 1774, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6), ([9470, 4347, 9383, 8171, 2784, 4109, 1226, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 7, [6264, 1, 3, 3, 3, 3, 3, 3, 3, 3, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2), ([2514, 12333, 9383, 888, 5659, 6442, 1226, 3913, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 8, [6124, 12151, 1121, 8539, 6262, 11668, 3251, 9383, 11002, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 10), ([1436, 1589, 6124, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        3, 3, 3, 3, 3, 3, 3], 3, [1505, 4101, 5211, 5393, 12908, 12182, 9368, 8434, 4768, 6124, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3], 12), ([6208, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 1, [6220, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2), ([4229, 11166, 4194, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3, [631, 5218, 3968, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([6456, 801, 7238, 6124, 4437, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 5, [6262, 4437, 10131, 6367, 548, 6367, 11056, 10353, 801, 7238, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 11), ([8472, 6115, 5479, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3, [4229, 4229, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([8171, 5698, 8831, 9383, 7616, 10025, 9383, 3913, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 8, [4136, 3968, 1388, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([12043, 8834, 7760, 9418, 3913, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 5, [7760, 9383, 1388, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([7238, 190, 3, 3, 3, 3, 3, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2, [9720, 9835, 7282, 1226, 1388, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 6), ([11723, 3918, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2, [6232, 9383, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([339, 425, 4580, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3, [9748, 637, 3913, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        3], 4), ([673, 9282, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 2, [673, 9383, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
        3, 3, 3, 3], 4), ([4913, 9383, 1121, 7412, 1226, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 5, [7616, 9383, 3968, 8204, 7856, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 7), ([3738, 2921, 1077, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3, [673, 6161, 4107, 
        1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 4), ([11637, 11849, 8834, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3, [1133, 1282, 11849, 4107, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 5), ([6124, 7760, 4111, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3, [12660, 4768, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3), ([12033, 6367, 10911, 427, 3684, 12211, 1133, 197, 11655, 12211, 11343, 3913, 3, 3, 3, 3, 3, 3, 3, 3], 12, [7616, 9383, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 3)]      
        '''
        batch = zip(*batch) # 将元组解压成列表， 第一个列表就是batch，第二列是其长度，第三列是答案的列表，第四列是答案长度
        # print("[DEBUG1]", list(batch))
        batch = [np.asarray(x) for x in batch]
        # print("[DEBUG2]", batch) # 此处打印所有的问句的编码向量
        # pdb.set_trace() 
        return batch   # 返回数组的格式[batch_size,4],4列分别为encoded_q, q_len, encoded_a, a_len

    def transform_sentence(self, sentence):
        """
        将句子转化为索引
        :param sentence:
        :return:
        """
        res = []

        # for word in jieba.lcut(sentence):  # 未分词，直接按单字处理
        #     res.append(self.func_word2index(word))
        for w in sentence:
            res.append(self.func_word2index(w))
        return res

    def transform_indexs(self, indexs):   # RestfulAPI中调用
        """
        将索引转化为句子,同时去除填充的标签
        :param indexs:索引序列
        :return:
        """
        res = []
        for index in indexs:
            if index == self.START_INDEX or index == self.PAD_INDEX \
                    or index == self.END_INDEX or index == self.UNK_INDEX:
                continue
            res.append(self.func_index2word(index))
        return ''.join(res)

    def _fit_data_(self):
        """
        得到处理后语料库的所有词，并将其编码为索引值
        :return:
        """
        if not os.path.exists(self.word2index_path):
            vocabularies = [x for x in self.data]  # x[0]为question,x[1]为answer
            self._fit_word_(itertools.chain(*vocabularies))   # itertools.chain可以把一组迭代对象串联起来，形成一个更大的迭代器;最终的格式为["q1","a1","q2","a12"......]
            with open(self.word2index_path, 'wb') as fw:
                pickle.dump(self.word2index, fw)
                print("[DEBUG] build W2ID succeed！")
        else:
            with open(self.word2index_path, 'rb') as fr:
                self.word2index = pickle.load(fr)
            self.index2word = dict([(v,k) for k,v in self.word2index.items()])  # k为值，v 为索引  index2word例如('1',你)
        self.vocab_size = len(self.word2index)
        # print("vocab_size = ", self.vocab_size, "word2index = ", self.word2index)
    
    # 导入word2id和id2word
    def load_word2id(self):
        with open(self.word2index_path, 'rb') as fr:
            word2id = pickle.load(fr)
        id2word = dict([(v, k) for k,v in word2id.items()])
        return word2id, id2word


    def loadDataset(self, train_path=None, vocab_size=30000):
        with codecs.open("search_dialog/data/vocab.txt", "r", "utf-8") as rfd:
            # word2id = {"<pad>": 0, "<go>": 1, "<eos>": 2, "<unknown>": 3}
            # id2word = {0: "<pad>", 1: "<go>", 2: "<eos>", 3: "<unknown>"}
            word2id = {"<SOS>": 0, "<EOS>": 1, "<UNK>": 2, "<PAD>": 3}
            id2word = {0: "SOS", 1: "<EOS>", 2: "<UNK>", 3: "<PAD>"}
        
            cnt = 4
            vocab_data = rfd.read().splitlines()
            for line in vocab_data[:vocab_size]:
                word, times = line.strip().split("\t")

                word2id[word] = cnt
                id2word[cnt] = word
                cnt += 1
            
        if train_path:
            with codecs.open(train_path, "r", "utf-8") as rfd:
                data = rfd.read().splitlines()
                data = [line.split("\t") for line in data]
                trainingSamples = [[text2id(item[0], word2id), 
                    text2id(item[1], word2id)] for item in data]
                # print("调试1 trainingSamples：", trainingSamples)  
                trainingSamples.sort(key=lambda x: len(x[0]))
                trainingSamples = [item for item in trainingSamples if 
                    (len(item[0]) >= 1 and len(item[1]) >= 1)]

                # print("调试2 trainingSamples：", trainingSamples)                
            print ("Load traindata form %s done." % train_path)
            return word2id, id2word, trainingSamples
        else:
            return word2id, id2word
        
    
    # 清洗每一个对话语料，并返回整个文件
    def load_data(self):
        """
        获取处理后的语料库
        :return:
        """
        if not os.path.exists(self.processed_path):
            data = self._extract_data()
            with open(self.processed_path, 'wb') as fw:
                pickle.dump(data, fw)
        else:
            with open(self.processed_path, 'rb') as fr:
                data = pickle.load(fr)
        # 根据CONFIG文件中配置的最大值和最小值问答对长度来进行数据过滤
        data = [x for x in data if self.min_q_len <= len(x[0]) <= self.max_q_len and self.min_a_len <= len(x[1]) <= self.max_a_len]
        return data

    def func_word2index(self, word):
        """
        将词转化为索引
        :param word:
        :return:
        """
        return self.word2index.get(word, self.word2index[self.UNK])  # 未知字符转换为self.UNK的索引

    def func_index2word(self, index):
        """
        将索引转化为词
        :param index:
        :return:
        """
        return self.index2word.get(index, self.UNK)  # 未知字符用UNK代替

    def jieba_sentence(self,vocabularies):
        groups = []
        for voc in vocabularies:
            groups.append(jieba.lcut(voc))

        return groups

    def _fit_word_(self, vocabularies, min_count=3, max_count=None):
        """
        将词表中所有的词转化为索引，过滤掉出现次数少于3次的词
        :param vocabularies:词表
        :return:
        """
        count = {}
        groups = self.jieba_sentence(vocabularies)
        for arr in groups:
            for a in arr:
                if a not in count:
                    count[a] = 0
                count[a] += 1

        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}

        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}

        index2word = [self.START] + [self.END] + [self.UNK] + [self.PAD] + [w for w in sorted(count.keys())]   # x[0]代表字符，x[1]代表出现的次数
        self.word2index = dict([(w, i) for i, w in enumerate(index2word)])  # i代表顺序，作为索引；w为字符   word2index例如 ('你'，1)
        self.index2word = dict([(i, w) for i, w in enumerate(index2word)])  # index2word ('1'，你)
        # print(self.word2index)

    # 把其中的word2id中的所有词全部按照id对照其词向量编码
    def emb_write(self):
        if not os.path.exists(self.emb_path):
            # word_vec = pickle.load(open(self.word_vec_path, 'rb'))
            word_vec = models.KeyedVectors.load_word2vec_format(
                self.word_vec_path, binary=False)

            emb = np.zeros((len(self.word2index), len(word_vec["的"])))
            print("[DEBUG] word2id长度: ", len(self.word2index))
            np.random.seed(1) # 设置随机数种子，那么只要参数一样，我们下次用同样的种子就能取出与之前相同的随机数。
            for word, id in self.word2index.items():
                if word in word_vec:
                    emb[id] = word_vec[word]
                else:
                    emb[id] = np.random.random(size=(300,)) - 0.5
            with open(self.emb_path, 'wb') as ep:
                pickle.dump(emb, ep)


    def _regular_(self, sen):
        """
        句子规范化，主要是对原始语料的句子进行一些标点符号的统一
        :param sen:
        :return:
        """
        sen = sen.replace('/', '')  # 将语料库中的/替为空
        sen = re.sub(r'…{1,100}', '…', sen)
        sen = re.sub(r'\.{3,100}', '…', sen)
        sen = re.sub(r'···{2,100}', '…', sen)
        sen = re.sub(r',{1,100}', '，', sen)
        sen = re.sub(r'\.{1,100}', '。', sen)
        sen = re.sub(r'。{1,100}', '。', sen)
        sen = re.sub(r'\?{1,100}', '？', sen)
        sen = re.sub(r'？{1,100}', '？', sen)
        sen = re.sub(r'!{1,100}', '！', sen)
        sen = re.sub(r'！{1,100}', '！', sen)
        sen = re.sub(r'~{1,100}', '～', sen)
        sen = re.sub(r'～{1,100}', '～', sen)
        sen = re.sub(r'[“”]{1,100}', '"', sen)
        sen = re.sub('[^\w\u4e00-\u9fff"。，？！～·]+', '', sen)  # 把非汉字字符和全角字符替换为空
        sen = re.sub(r'[ˇˊˋˍεπのゞェーω]', '', sen)

        return sen

    def _good_line_(self, line):
        """
        判断一句话是否是好的语料,即判断汉字所占比例>=0.8，且数字字母<3个，其他字符<3个
        :param line:
        :return:
        """
        if len(line) == 0:
            return False
        ch_count = 0
        for c in line:
            # 中文字符范围
            if '\u4e00' <= c <= '\u9fff':
                ch_count += 1
        if ch_count / float(len(line)) >= 0.8 and len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) < 3 \
                and len(re.findall(r'[ˇˊˋˍεπのゞェーω]', ''.join(line))) < 3:
            return True
        return False

    def _extract_data(self):
        res = []
        for maindir, subdir, file_name_list in os.walk(self.path):
            for filename in file_name_list:
                with open(self.path+filename, 'r', encoding='utf-8') as fr:
                    for line in fr:
                        if '\t' not in line:  # 错误处理的句子进行过滤
                            continue
                        q, a = line.split('\t')
                        q = self._regular_(q)
                        a = self._regular_(a)
                        if self._good_line_(q) and self._good_line_(a):
                            res.append((q, a))

        return res

    def __len__(self):  # 函数的重载
        """
        返回处理后的语料库中问答对的数量
        :return:
        """
        return len(self.data)

def loadDataset(train_path=None, vocab_size=10000):
    with codecs.open(vocab_path, "r", "utf-8") as rfd:
        word2id = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        id2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        cnt = 4
        vocab_data = rfd.read().splitlines()
        for line in vocab_data[:vocab_size]:
            word, _ = line.strip().split("\t")
            word2id[word] = cnt
            id2word[cnt] = word
            cnt += 1

    if train_path:
        with codecs.open(train_path, "r", "utf-8") as rfd:
            data = rfd.read().splitlines()
            data = [line.split("\t") for line in data]
            trainingSamples = [[text2id(item[0], word2id), 
                text2id(item[1], word2id)] for item in data]
            # print("调试1 trainingSamples：", trainingSamples)  
            trainingSamples.sort(key=lambda x: len(x[0]))
            trainingSamples = [item for item in trainingSamples if 
                (len(item[0]) >= 1 and len(item[1]) >= 1)]

            # print("调试2 trainingSamples：", trainingSamples)                
        print ("Load traindata form %s done." % train_path)
        return word2id, id2word, trainingSamples
    else:
        return word2id, id2word

if __name__ == "__main__":
    du = DataUnit(**data_config)
    print("[DEBUG] len(du) = ", len(du))
    # du = DataUnit(**data_config)
    # w2id, id2w = du.load_word2id()
    # print(du._regular_("hahahh,,,,a3!!!@#$%^&*():>?/*&……5/…………///4```······#@$%@@#%^^^……………… 安抚"))
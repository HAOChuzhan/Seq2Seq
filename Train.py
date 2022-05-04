
# encoding: utf-8

import DataProcessing
from DialogManage import DialogManagement
import os
import codecs
import tensorflow as tf
from SequenceToSequence import Seq2Seq
from tqdm import tqdm
import numpy as np
from CONFIG import BASE_MODEL_DIR, MODEL_NAME, data_config, model_config, \
    n_epoch, batch_size, keep_prob
import pickle

# 是否在原有模型的基础上继续训练
continue_train = True


def train():
    """
    训练模型
    :return:
    """
    du = DataProcessing.DataUnit(**data_config) # ** 的作用：** 会以键/值的形式解包一个字典，使其成为一个独立的关键字参数
    save_path = os.path.join(BASE_MODEL_DIR, MODEL_NAME)
    steps = int(len(du) / batch_size) + 1   # len(du)处理后的语料库中问答对的数量
    print("[DEBUG] steps:", steps, "len(du):", len(du))
    # tf.test.is_built_with_cuda() # 是否能够使用GPU进行运算
    # 创建session的时候设置显存根据需要动态申请
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 当使用GPU时，设置GPU内存使用最大比例
    config.gpu_options.allow_growth = True # 当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存

    with tf.Graph().as_default(): # 返回值：返回一个上下文管理器，这个上下管理器使用这个图作为默认的图
        with tf.Session(config=config) as sess:
            # 定义模型
            model = Seq2Seq(batch_size=batch_size,
                            encoder_vocab_size=du.vocab_size,
                            decoder_vocab_size=du.vocab_size,
                            mode='train',
                            **model_config)

            init = tf.global_variables_initializer()
            writer=tf.summary.FileWriter('./graph/nlp',sess.graph) # 保存整个图中变量的名称及值
            sess.run(init)
            if continue_train:
                model.load(sess, save_path)

            emb = pickle.load(open(data_config["emb_path"], 'rb'))
            model.feed_embedding(sess, encoder=emb,decoder=emb)

            for epoch in range(1, n_epoch + 1):
                costs = []
                bar = tqdm(range(steps), total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))   #进度条
                for _ in bar:
                    x, xl, y, yl = du.next_batch(batch_size)  # x为question,xl为question实际长度；y为answer,yl为answer实际长度
                    max_len = np.max(yl)  # 实际最长句子的长度
                    y = y[:, 0:max_len]  # 表示所有行的第0:max_len列
                    cost, lr = model.train(sess, x, xl, y, yl, keep_prob)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(epoch, np.mean(costs), lr))
                model.save(sess, save_path=save_path)

# 测试seq2seq：
def chatbot_api():
    with codecs.open("scripts/questions.txt", "r", encoding="utf-8") as rf1, \
        codecs.open("scripts/answers.txt", "w", encoding="utf-8") as rw:
        questions = rf1.readlines()
        print(len(questions))
        for i in range(len(questions)):
            info1 = questions[i].strip()
            response = DialogManagement.process_dialog(info1, use_task=False)
            rw.write(response + "\n")
            print(i)
            if i == 500: break
    return "完成生成50W答复！"


if __name__ == '__main__':
    train()
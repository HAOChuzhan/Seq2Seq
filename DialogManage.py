
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import DataProcessing
import tensorflow as tf
import numpy as np
from SequenceToSequence import Seq2Seq
from utils.nlp_util import NlpUtil
from search_dialog.search_core import SearchCore
from CONFIG import BASE_MODEL_DIR, MODEL_NAME, data_config, model_config
from utils.tools import word_count

from task_dialog.task_core import TaskCore
from utils.contextual_fusion.predict import PredictManager

class DialogStatsu(object):

    def __init__(self):
        self.intent = None
        self.state = None # 1为chitchat，2为commerce主题 
        self.theme = None # 1.日常 2.电影 3.百科
        # Special slots
        self.start_flag = 0 # 0是未开启对话，1是正在对话状态中
        self.query_intent = None
        # Time
        self.start_time = None # 对话开始时间
        self.end_time = None # 对话结束时间

        # Dialog context
        self.context = []

class DialogManagement(object):

    dialog_status = DialogStatsu()
    
    
    # context = manager.predict_result("今 天 可 以 发 货 吗	    那 明 天 呢")
    # print(context) # 那 明 可 以 发 货 天 呢

    @classmethod
    def _predict_via_seq2seq(cls, msg):
        du = DataProcessing.DataUnit(**data_config)

        save_path = os.path.join(BASE_MODEL_DIR, MODEL_NAME)
        # print("[DEBUG] model address = ", save_path)
        batch_size = 1
        tf.reset_default_graph()
        model = Seq2Seq(batch_size=batch_size,
                        encoder_vocab_size=du.vocab_size,
                        decoder_vocab_size=du.vocab_size,
                        mode='decode',
                        **model_config)
        # 创建session的时候允许显存增长
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            model.load(sess, save_path)

            indexs = du.transform_sentence(msg) # 把句子转换为索引
            print("[DEBUG] indexs =", indexs)
            x = np.asarray(indexs).reshape((1, -1))  # 转为1行
            xl = np.asarray(len(indexs)).reshape((1,))
            pred = model.predict(
                sess, np.array(x),
                np.array(xl)
            )
            response = du.transform_indexs(pred[0])  # 将索引转为句子
            return response


    @classmethod
    def process_dialog(cls, msg, use_task=True):
        if use_task:
            task_response, cls.dialog_status = TaskCore.task_handle(msg, cls.dialog_status)# "这里是任务式回复语句"
        else:
            task_response = None
        # Search response
        if len(cls.dialog_status.context) >= 3 and word_count(msg) <= 4:
            print("[DEBUG] dialog_status.context =", cls.dialog_status.context)
            user_msgs = cls.dialog_status.context[::2][-3:]
            # msg = "<s>".join(user_msgs)
        
            manager = PredictManager("utils/contextual_fusion/pretrained_weights/rewrite.tar.gz")
            msg = manager.predict_result(' '.join(list(user_msgs[-2]))+"\t"+' '.join(list(user_msgs[-1])))
            msg = msg.replace(" ", "")
            print("[DEBUG] Contextual =",msg)
            mode = "cr"
        else:
            mode = "qa"
        # tokenize函数
        msg_tokens = NlpUtil.tokenize(msg, filter_punctuations=False, filter_stopwords=False, filter_alpha=False)
        print("[DEBUG] MsgTokenize =", msg_tokens)
        # Search response
        search_response, sim_scores = SearchCore.search(msg_tokens, mode=mode)
        
        if task_response:
            response = task_response
        elif sim_scores>=5.0:
            response = search_response
        else:
            seq2seq_response = cls._predict_via_seq2seq(msg_tokens)
            print("[DEBUG] seq2seq_response =", seq2seq_response)
            response = seq2seq_response

        return response



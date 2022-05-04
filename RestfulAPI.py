# encoding: utf-8

import os
import codecs
from random import random
from datetime import datetime
import tensorflow as tf
import numpy as np

from SequenceToSequence import Seq2Seq
import DataProcessing
from DialogManage import DialogManagement
from CONFIG import BASE_MODEL_DIR, MODEL_NAME, data_config, model_config
from theme_dialog.chitchat import chitchat_status

import pymysql
import tornado.web
import tornado.ioloop
from tornado.options import define,options
import tornado.httpserver
from tornado.web import RequestHandler

 

tornado.options.define("port",default=5005,help="run port", type=int)



def chat_mode(mode):

    while True:
        if mode == 'chitchat':
            return ["hello，欢迎来到闲聊模式！我是您的助理小Z","==== 提示：如果想要转换成电商主题模式，请输入 commerce ===="]
        elif mode == 'commerce':
            return ["欢迎来到电商咨询模式，我是您的私人助理小Z！","==== 提示：如果想要转换成闲聊模式，请输入 chitchat ===="]
        # else if mode == 'end':
        #     return "byebye~ 小詹期待和您下次再见！"


def chatbot_api(infos):
    
    while True:
        q = infos
        if q is None or q.strip() == '':
            return "请输入聊天信息"
            continue
        # 获取对话开始的时间
        if DialogManagement.dialog_status.start_flag == 0:
            DialogManagement.dialog_status.start_time = datetime.now()
            DialogManagement.dialog_status.start_flag = 1

        q = q.strip()
        state = DialogManagement.dialog_status.state
        print("[DEBUG] state =", state," ","theme =", DialogManagement.dialog_status.theme)
        if q == "chitchat" and state != 1:
            DialogManagement.dialog_status.state = 1
            return ["hello，欢迎来到闲聊模式！我是您的助理小Z","<br>","可以想我问任何的问题哟，例如：<br>想知道天气请输入：南京天气，不开心的时候输入：笑话 ","<br>","=============== 提示：===============<br>","如果想要转换成电商客服模式，请输入 commerce"]
        elif q == "chitchat" and state == 1:
            return ["亲亲，现在已经是闲聊模式了哦，请勿重复输入 chitchat"]
            # self.write("亲亲，现在已经是闲聊模式了哦，请勿重复输入 chitchat")
        elif q == "commerce" and state != 2:
            DialogManagement.dialog_status.state = 2
            return ["欢迎来到电商咨询模式，我是您的私人助理小Z！","<br>","=============== 提示：===============<br>","如果想要转换成闲聊模式，请输入 chitchat"]
        elif q == "commerce" and state == 2:
            return ["亲亲，现在已经是电商咨询主题了哦，请勿重复输入 commerce"]
        elif q == "end":
            DialogManagement.dialog_status.end_time = datetime.now()
            DialogManagement.dialog_status.state = None
            DialogManagement.dialog_status.theme = None
            DialogManagement.dialog_status.start_flag = 0
            return ["byebye~ 小Z期待与您下次再见！"]
        elif q != "chitchat" and q != "commerce" and state == 2:
            DialogManagement.dialog_status.context.append(q)
            response = DialogManagement.process_dialog(q, use_task=True)
            DialogManagement.dialog_status.context.append(response)
            return [str(response)]
        elif q != "chitchat" and q != "commerce" and state == 1: #and DialogManagement.dialog_status.theme == None
            response = chitchat_status(q)
            return response
           
            # if q == "1":
            #     DialogManagement.dialog_status.theme = 1
            #     return ["现在已进入日常主题，我是你的朋友小Z哟，接下来我们一起聊天吧！"]
            # elif q == "2":
            #     DialogManagement.dialog_status.theme = 2
            #     return ["现在已进入电影主题，我们一起来探讨泰坦尼克号的神秘吧！"]
            # elif q == "3":
            #     DialogManagement.dialog_status.theme = 3
            #     return ["现在已进入百科主题，有什么不懂得尽管吩咐小的✔"]
                     
            # else: # 请输入主题编号
            #     # res = chat_theme(q, DialogManagement.dialog_status.theme)
            #     return ["请输入主题编号：1/2/3"]

        # elif q != "chitchat" and q != "commerce" and state == 1 and DialogManagement.dialog_status.theme != None:
            
        #     response = chitchat_status(q)
        #     return  response # ["现在已进入主题对话模式"] # 这里是调用对用的函数来产生回复
        else:
            return ["请提出符合规范的问题哟(＾Ｕ＾)"]



class BaseHandler(RequestHandler):
    """解决JS跨域请求问题"""

    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET')
        self.set_header('Access-Control-Max-Age', 1000)
        self.set_header('Access-Control-Allow-Headers', '*')
        # self.set_header('Content-type', 'application/json')


class IndexHandler(BaseHandler):
    # 添加一个处理get请求方式的方法
    def get(self):
        # 向响应中，添加数据
       
        infos = self.get_query_argument("infos")
        print("Q:", infos)
        
        # 捕捉服务器异常信息
        try:
            result = chatbot_api(infos=infos)
            for i in range(len(result)):
                self.write(result[i])
            print("Z:", "".join(result))
            print("[DEBUG] start_flag:", DialogManagement.dialog_status.start_flag)

            if DialogManagement.dialog_status.start_flag == 0:
                # 然后我们开始将对话数据存储到我们Mysql数据库中
                # SQL语句
                print("这一段session将被储存到MySQL数据库中")
                start = DialogManagement.dialog_status.start_time
                end = DialogManagement.dialog_status.end_time
                # 这里把对话内容转化成json格式
                content = dict()
                content['name'] = "War11"
                #  DialogManagement.dialog_status.context[]

                emotion = ['中性', '积极', '消极', '十分消极']
                emotion_score = random()
                
                sql = "insert into chatlog(content,emotion,emotion_score,start_time,end_time) values('{}','{}',{},'{}','{}')".format('content: Q: 你好, A: 你好你好亲亲', '、'.join(emotion), emotion_score, start, end)
                print(sql)
                # 创建连接对象，填写相应的参数 主机ip、端口、数据库的用户名、密码、数据库名字
                conn = pymysql.connect(host='127.0.0.1', port=3306, passwd='135792468',user='root', db='war')
                # 创建连接游标,用于读写操作
                cursor = conn.cursor()
                # 执行sql语句，并返回影响行数
                effect_row = cursor.execute(sql)
                # 提交数据
                conn.commit()
                DialogManagement.dialog_status.start_time = None
                DialogManagement.dialog_status.end_time = None
                cursor.close()
                conn.close()
        except:
            result = "不太能理解您说的话哟"
            self.write(result)
        
        
class IndexHandler2(BaseHandler):
    def get(self):
        mode = self.get_query_argument("infos")
        res = chat_mode(mode)

        self.write(res[0])
        self.write("<br>")
        self.write(res[1])



if __name__ == '__main__':
    
    # 创建应用对象
    app = tornado.web.Application(
        handlers = [                                                         # handlers 固定的变量命名规则
            (r'/api/chatbot', IndexHandler),
            (r"/api/chatmode",chat_mode),
            # (r"/sub/(.+)/([0-9]+)",SubHandler),                            # URL传参,正则匹配      点号. -> 除自身之外
            # (r"/use/(?P<name>.+)/(?P<age>[0-9]+)",UseSubHandler),          # 按照名字传入参数
        ],
        # template_path = 'templates',                                       # 模板路径（这里的路径根据自己的来）
        # debug=True                                                         # 代码保存上次之后，立即重启服务器，就不用手动重启了
    )
    tornado.options.parse_command_line()
    # http_server = tornado.httpserver.HTTPServer(application)
    # 设置监听的端口号
    app.listen(options.port)
    # 启动web程序，开始监听端口的连接
    tornado.ioloop.IOLoop.current().start()

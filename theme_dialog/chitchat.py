import urllib.parse
import urllib.request
import json
import re
'''
天气：msg=天气深圳             中英翻译：msg=翻译i love you
智能聊天：msg=你好             笑话：msg=笑话
歌词⑴：msg=歌词后来           歌词⑵：msg=歌词后来-刘若英  
计算⑴：msg=计算1+1*2/3-4      计算⑵：msg=1+1*2/3-4
ＩＰ⑴：msg=归属127.0.0.1      ＩＰ⑵：msg=127.0.0.1
手机⑴：msg=归属13430108888    手机⑵：msg=13430108888
成语查询：msg=成语一生一世      五笔/拼音：msg=好字的五笔/拼音
'''

url = "http://api.qingyunke.com/api.php"
values = dict()
values['key'] = 'free'
values['appid'] = '0'
# pattern = re.compile(r"{br}")

def chitchat_status(msg):
    values['msg'] = msg
    # 把key-value这样的键值对转换成a=1&b=2这样的字符串
    data = urllib.parse.urlencode(values)

    req = url + '?' + data
    response = urllib.request.urlopen(req)
    # print(type(response))  # 打印<class 'http.client.HTTPResponse'>
    # 打印Http状态码
    # print(response.status) # 打印200
    # 读取服务器返回的数据,对HTTPResponse类型数据进行读取操作
    the_page = response.read()
    # 中文编码格式打印数据
    print(the_page.decode("utf-8"))

    res = dict()
    res = json.loads(the_page.decode("utf-8"))
    print(res["content"])
    res = re.sub(r"{br}", "<br>", res["content"])
    print(type(the_page.decode("utf-8")))
    return res
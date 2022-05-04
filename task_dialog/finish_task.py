import re
import random
from utils.tools import word_count

finish_pattern = re.compile(
    r"谢|"
    r"再见|"
    r"拜拜|"
    r"没.*了")

def intent_update(msg, dialog_status):
    if word_count(msg)<=8 and finish_pattern.search(msg):
        print()
        dialog_status.intent = "finish"
    return dialog_status

def finish_handle(msg, dialog_status):
    goodbye_sheet = [
        "天气多变，请注意添衣减衣吖，若有怠慢的地方，您请多多包含，期待再次为您服务", 
        "亲爱哒，有什么不明白随时联系我哦，很荣幸能够为您服务呢",
        "亲爱哒，感谢您对小店的支持，祝您生活愉快~",
        "祝福您幸福快乐，前程锦绣，还请您点击表情栏旁边的“+”打赏我一个评价哦"]
    response = random.sample(goodbye_sheet, 1)[0]
    return response
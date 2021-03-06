import re
import random
from utils.tools import word_count

ch_pattern = re.compile(r"[\u4e00-\u9fa5]+") # 包含所有的汉字
not_match_pattern = re.compile(r"吗|\?|？|多|哪|怎|什么|啥|退|发票")
match_pattern = re.compile(r"好|哦|嗯|哈|麻烦|谢|括号|\d+")
bracket_pattern = re.compile(r"\[.*\]")

def intent_update(msg, dialog_status):
    msg = bracket_pattern.sub("括号", msg)
    if word_count(msg)<=4 and match_pattern.search(msg):
        if not ch_pattern.search(msg) or not not_match_pattern.search(msg):
            dialog_status.intent = "short_query"
    return dialog_status 


def short_query_handle(msg, dialog_status):
    responses = [
        "好的，亲爱哒，还有什么问题我可以帮您呢? (^_^)"
    ] 
    response = random.sample(responses, 1)[0]
    return response
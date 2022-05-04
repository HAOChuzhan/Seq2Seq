import re


# 计算中文汉字数
def word_count(msg):
    ch_pattern = re.compile(r"[\u4e00-\u9fa5]")
    remove_pattern = re.compile(r"好的|谢谢|感谢")
    msg = remove_pattern.sub("", msg)
    res = ch_pattern.findall(msg)
    length = len("".join(res))
    return length
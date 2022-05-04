from torch.utils.data import Dataset
import torch
import codecs

with codecs.open("./data/xiaohuangji.txt", 'r', encoding="utf-8") as fr, \
    codecs.open("./data/questions.txt", 'w', encoding="utf-8") as fw1, \
    codecs.open("./data/answers.txt", 'w', encoding="utf-8") as fw2, \
    codecs.open("./data/xiaohuangji_filter.txt.", 'w', encoding="utf-8") as fw3:

    
    cnt = 0
    for line in fr:
        res = line.strip().split('\t')
        if len(res)!=2:
            continue
        q = res[0]
        a = res[1]
        fw3.write(q+'\t'+a+'\n')
        cnt += 1
        print(cnt)
    print(cnt)

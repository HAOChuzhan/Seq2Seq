import paddlehub as hub
import os
# 加载模型
senta = hub.Module(name="senta_bilstm")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def sentiment_analysis(sentence):
    # 待分类文本
    test_text = [
        sentence,
    ]
    # 情感分类
    results = senta.sentiment_classify(data={"text": test_text})  
    # 得到结果
    return results 

if __name__ == "__main__":
    res = sentiment_analysis("这个东西还可以")
    print(res)
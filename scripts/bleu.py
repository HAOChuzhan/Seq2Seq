import codecs
import sys
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def bleu(answersFilePath, standardAnswersFilePath):
    with codecs.open(answersFilePath, 'r', "utf-8") as rf_answers:
        with codecs.open(standardAnswersFilePath, 'r', "utf-8") as rf_standardAnswers:
            score = []
            answersFile = rf_answers.readlines()[:500]
            standardAnswersFile = rf_standardAnswers.readlines()[:500]

            chencheery = SmoothingFunction()
            for i in range(len(answersFile)):
                candidate = jieba.lcut(answersFile[i].strip().split()[0])
                reference = jieba.lcut(standardAnswersFile[i].strip())
                # candidate = "我是中人。"
                # reference = "我是中国人。"
                print("生成答复：{}".format(candidate))
                print("标准答复：{refer}".format(refer = reference))
                bleu_score = sentence_bleu([reference], candidate, weights=(1,0,0,0),smoothing_function=chencheery.method1)# weight = 0.35,0.45,0.1,0.1
                print(bleu_score)
                score.append(bleu_score)

            precisionScore = round(sum(score)/len(answersFile), 6)
            return precisionScore         

if __name__ == "__main__":
    candidateFile = sys.argv[1]
    referenceFile = sys.argv[2]
    Socre = bleu(candidateFile, referenceFile)
    print("[总均分]:", Socre)
    '''
    python bleu.py answers.txt standardAnswers.txt
    [总均分]: 0.404239
    '''
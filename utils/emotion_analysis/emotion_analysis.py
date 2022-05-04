import codecs
import jieba
from collections import defaultdict
import os
os.chdir("D:\\AI\\任务型对话系统\\对话系统\\Smart-Customer-chatbot\\utils")
# 建立一个去除否定词和程度副词的停用词词典
def create_newstopwords():
    stopwords = set()
    with codecs.open('conf/stopwords.txt', 'r', encoding='utf-8') as fr1:
        for word in fr1:
            stopwords.add(word.strip())
    
    with codecs.open('conf/negative_words.txt','r',encoding='utf-8') as fr2:
        not_word_list = fr2.readlines()
        not_word_list = [w.strip() for w in not_word_list]

    with codecs.open('conf/adverbs_degree.txt', 'r', encoding='utf-8') as fr3:
        degree_list = fr3.readlines()
        degree_list = [w.split(',')[0] for w in degree_list]

    with codecs.open('conf/stopwords_new.txt', 'w', encoding='utf-8') as fw:
        for word in stopwords:
            if (word not in not_word_list) and (word not in degree_list):
                fw.write(word+'\n')
# 分词后去除停用词
def seg_word(sentence):
    seg_list = jieba.cut(sentence)
    seg_result = [] 
    for w in seg_list:
        seg_result.append(w)
    stopwords = set()
    with codecs.open("conf/stopwords.txt", 'r', encoding="utf-8") as fr:
        for line in fr:
            stopwords.add(line.strip())
    return list(filter(lambda x:x not in stopwords, seg_result))

# 找出文本中的情感词、否定词、和程度副词
def classify_words(word_list):
    #读取情感词典文件
	sen_file = codecs.open('conf/BosonNLP_sentiment_score.txt','r+',encoding='utf-8')
	#获取词典文件内容
	sen_list = sen_file.readlines()
	#创建情感字典
	sen_dict = defaultdict()
	#读取词典每一行的内容，将其转换成字典对象，key为情感词，value为其对应的权重
	for i in sen_list:
		if len(i.split(' '))==2:
			sen_dict[i.split(' ')[0]] = i.split(' ')[1]
 
	#读取否定词文件
	not_word_file = codecs.open('conf/negative_words.txt','r+',encoding='utf-8')
	not_word_list = not_word_file.readlines()
	#读取程度副词文件
	degree_file = codecs.open('conf/adverbs_degree.txt','r+', encoding='utf-8')
	degree_list = degree_file.readlines()
	degree_dict = defaultdict()
	for i in degree_list:
		degree_dict[i.split(',')[0]] = i.split(',')[1]

	sen_word = dict()
	not_word = dict()
	degree_word = dict()
	#分类
	for i in range(len(word_list)):
		word = word_list[i]
		if word in sen_dict.keys() and word not in not_word_list and word not in degree_dict.keys():
			# 找出分词结果中在情感字典中的词
			sen_word[i] = sen_dict[word]
		elif word in not_word_list and word not in degree_dict.keys():
			# 分词结果中在否定词列表中的词
			not_word[i] = -1
		elif word in degree_dict.keys():
			# 分词结果中在程度副词中的词
			degree_word[i]  = degree_dict[word]

	#关闭打开的文件
	sen_file.close()
	not_word_file.close()
	degree_file.close()
	#返回分类结果
	return sen_word,not_word,degree_word


#计算情感词的分数
def score_sentiment(sen_word,not_word,degree_word,seg_result):
	#权重初始化为1
	W = 1
	score = 0
	#情感词下标初始化
	sentiment_index = -1
	#情感词的位置下标集合
	sentiment_index_list = list(sen_word.keys())
	#遍历分词结果
	for i in range(len(seg_result)):
		#如果是情感词
		if i in sen_word.keys():
			#权重*情感词得分
			score += W*float(sen_word[i])
			#情感词下标加一，获取下一个情感词的位置
			sentiment_index += 1
			if sentiment_index < len(sentiment_index_list)-1:
				#判断当前的情感词与下一个情感词之间是否有程度副词或否定词
				for j in range(sentiment_index_list[sentiment_index],sentiment_index_list[sentiment_index+1]):
					#更新权重，如果有否定词，权重取反
					if j in not_word.keys():
						W *= -1
					elif j in degree_word.keys():
						W *= float(degree_word[j])	
		#定位到下一个情感词
		if sentiment_index < len(sentiment_index_list)-1:
			i = sentiment_index_list[sentiment_index+1]
	return score

#计算得分
def sentiment_score(sentence):
    #1.对文档分词
    seg_list = seg_word(sentence)
    #2.将分词结果转换成字典，找出情感词、否定词和程度副词
    sen_word,not_word,degree_word = classify_words(seg_list)
    print(sen_word,not_word, degree_word)
    #3.计算得分
    score = score_sentiment(sen_word,not_word,degree_word,seg_list)
    return score

if __name__ == "__main__":
    test_str = "你们的服务态度太差了"
	

    print(seg_word(test_str), sentiment_score(test_str))
    print(seg_word('我要投诉你们'),sentiment_score('我要投诉你们'))    



'''

######################################
         BosonNLP情感词典
######################################

BosonNLP情感词典是从微博、新闻、论坛等数据来源的上百万篇情感标注数据当中自动构建的情感极性词典。因为标注包括微博数据，该词典囊括了很多网络用语及非正式简称，对非规范文本也有较高的覆盖率。该情感词典可以用于构建社交媒体情感分析引擎，负面内容发现等应用。

在BosonNLP情感词典中，文本采用UTF-8进行编码，每行为一个情感词及其对应的情感分值，以空格分隔，共包括114767个词语。其中负数代表偏负面的词语，非负数代表偏正面的词语，正负的程度可以由数值的大小反应出。

格式:
[[词语]] [[情感值]]

例：
最尼玛 -6.70400012637
扰民 -6.49756445867
fuck... -6.32963390433
RNM -6.21861284426
wcnmlgb -5.96710044003
2.5: -5.90459648251
Fxxk -5.87247473641
MLP -5.87247473641
吃哑巴亏 -5.77120419579

来源：
http://bosonnlp.com
'''
import time
import codecs
from utils.bm25_util import BM25Util 

DEBUG_MODE = True

class BM25Model(object):
    def __init__(self, corpus_file, word2id):
        time_s = time.time()

        size = 500000 if DEBUG_MODE else 10000000
        with codecs.open(corpus_file, 'r', "utf-8") as rfd:
            data = [s.strip().split("\t") for s in rfd.readlines()[:size]] 
           
            self.contexts = [[w for w in q.split() if w in word2id] for q, _ in data]
            self.responses = [a.replace(" ", "") for _, a in data]
        self.bm25_instance = BM25Util(self.contexts)

        print("Time to build BM25 model:%.4f seconds." % (time.time()-time_s))


    def similarity(self, query, size=10):
        return self.bm25_instance.similarity(query, size)

    def get_doc(self, sim_items):
        docs = [self.contexts[id_] for id_, score in sim_items]
        answers = [self.responses[id_] for id_, score in sim_items]
        return docs, answers
   
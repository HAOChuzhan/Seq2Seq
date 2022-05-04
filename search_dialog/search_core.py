# import sys
# sys.path.append("..")

from search_dialog import search_config
from search_dialog.bm25 import BM25Model
from DataProcessing import loadDataset, DataUnit
from CONFIG import data_config

SEARCH_NODEL = "bm25"


class SearchCore(object):
    data_inst = DataUnit(**data_config)
    word2id, _ = data_inst.loadDataset(vocab_size=30000)
    
    
    if SEARCH_NODEL == "bm25":
        qa_search_inst = BM25Model(search_config.question_answer_path, word2id=word2id)
        cr_search_inst = BM25Model(search_config.context_response_path, word2id=word2id)


    @classmethod
    def search(cls, msg_tokens, mode='qa', filter_pattern=None):
        query = [w for w in msg_tokens if w in cls.word2id]
        search_inst = cls.qa_search_inst if mode == "qa" else cls.cr_search_inst
        sim_items = search_inst.similarity(query, size=10)
        docs, answers = search_inst.get_doc(sim_items)

        print("[DEBUG] 猜你想问: ", docs[:5])

        # if filter_pattern:
        #     new_docs, new_answers = [], []
        #     for doc,ans in zip(docs, answers):
        #         if not filter_pattern.search(ans):
        #             new_docs.append(doc)
        #             new_answers.append(ans)
        #     docs, answers = new_docs, new_answers

        print("[DEBUG] init_query = %s, filter_query = %s" % ("".join(msg_tokens), "".join(query)))
        response, score = answers[0], sim_items[0][1]
        print("[DEBUG] %s_search_sim_query = %s, score = %.4f" % (mode, "".join(docs[0]), score))
        if score <= 1.0:
            response, score = "亲爱哒，还有什么可以帮到您呢", 2.0
        print("[DEBUG] search_response =", response)
        return response, score


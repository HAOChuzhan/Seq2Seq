import os

def file_path(dirname, file_name):
    module_path = os.path.dirname(__file__)
    file_path = os.path.join(module_path, dirname, file_name)
    return file_path

word2index = "data/v1.w2v_sgns_win2_d300.kv"
index_path = file_path("index", "similarity")
question_answer_path = file_path("data", "question_answer.txt")
context_response_path = file_path("data", "context_response.txt")
vocab_path = file_path("data", "vocab.txt")
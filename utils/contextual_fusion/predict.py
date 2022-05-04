
# import sys
# import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
# WARNING: Do not exclude these imports
from utils.contextual_fusion.data_reader import RewriteDatasetReader
from utils.contextual_fusion.predictor import RewritePredictor
from utils.contextual_fusion.model import UnifiedFollowUp

# from data_reader import RewriteDatasetReader
# from predictor import RewritePredictor
# from model import UnifiedFollowUp



class PredictManager:

    def __init__(self, archive_file):
        archive = load_archive(archive_file)
        self.predictor = Predictor.from_archive(
            archive, predictor_name="rewrite")

    def predict_result(self, dialog_flatten: str):
        # dialog_flatten is split by \t
        dialog_snippets = dialog_flatten.split("\t")
        param = {
            "context": dialog_snippets[:-1],
            "current": dialog_snippets[-1]
        }
        restate = self.predictor.predict_json(param)["predicted_tokens"]
        return restate


if __name__ == '__main__':
    manager = PredictManager("pretrained_weights/rewrite.tar.gz")
    result =  manager.predict_result("今 天 可 以 发 货 吗	    那 下 周 呢") 
    # result = manager.predict_result("买 茶 叶 嘛		我 在 网 上 买		我 问 你 买 嘛")
    result1 = manager.predict_result("今 天 可 以 发 货 吗         今 天 暂 时 不 能	    那 大 后 天 呢")

    print(''.join(result.split()),"\n",result1)

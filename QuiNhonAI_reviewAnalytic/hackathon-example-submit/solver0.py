from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig, RobertaTokenizer
from processing import TextProcessing
from model import RobertaMultiHeadClassifier, RobertaEnsembleLayer, RobertaMixLayer
import torch


class ClassifyReviewSolver:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processing_text = None
        self.setup(model_path=config.MODEL_PATH)

    def setup(self, model_path):
        tokenizer = RobertaTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
        tokenizer.do_lower_case = True
        
        config = AutoConfig.from_pretrained(
            model_path,
        )

        self.model = RobertaEnsembleLayer.from_pretrained(
            model_path,
            config=config,
        )

        self.model.to(self.config.DEVICE)

        self.processing_text = TextProcessing(tokenizer=tokenizer,
                                              max_length_rating=self.config.MAX_LEN_RATING,
                                              max_length_detect=self.config.MAX_LEN_DETECTION)

    def solve(self, text):
        clean_text = self.processing_text.clean_sentence(text)
        tokenize_text = self.processing_text.tokenize_rate(clean_text)
        for key in tokenize_text.keys():
            tokenize_text[key] = torch.tensor([tokenize_text[key]]).to(self.config.DEVICE)

        self.model.eval()
        with torch.no_grad():
            output = self.model(**tokenize_text)
        prediction = torch.argmax(output.logits, dim=-1).cpu().numpy().tolist()
        
        return prediction

if __name__ == "__main__":
    import config
    from time import time
    RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]

    review_solver = ClassifyReviewSolver(config)

    review_sentence = "Nhà hàng ok"
    time1 = time()
    predict_results = review_solver.solve(review_sentence)
    time2 = time()

    print(time2 - time1)
    output = {
        "review": review_sentence,
        "results": {}
      }
    for count, r in enumerate(RATING_ASPECTS):
        output["results"][r] = predict_results[count]

    print(output)

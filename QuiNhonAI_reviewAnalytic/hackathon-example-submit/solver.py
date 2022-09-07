from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig, RobertaTokenizer
from processing import TextProcessing
from model import RobertaMultiHeadClassifier, RobertaMixLayer, RobertaEnsembleLayer
import torch


class ClassifyReviewSolver:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.detect_model = None
        self.processing_text = None
        self.setup(model_path=config.MODEL_PATH, detect_model_path=config.DETECT_MODEL_PATH)

    def setup(self, model_path, detect_model_path):
        tokenizer = RobertaTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
        tokenizer.do_lower_case = True
        
        config = AutoConfig.from_pretrained(
            model_path,
        )

        self.model = RobertaMixLayer.from_pretrained(
            model_path,
            config=config,
        )
        
        config = AutoConfig.from_pretrained(
            detect_model_path,
        )

        self.detect_model = AutoModelForSequenceClassification.from_pretrained(
            detect_model_path,
            config=config,
        )

        self.model.to(self.config.DEVICE)
        self.detect_model.to(self.config.DEVICE)

        self.processing_text = TextProcessing(tokenizer=tokenizer,
                                              max_length_rating=self.config.MAX_LEN_RATING,
                                              max_length_detect= self.config.MAX_LEN_DETECTION)

    def solve(self, text):
        clean_text = self.processing_text.clean_sentence(text)
        tokenize_text = self.processing_text.tokenize_rate(clean_text)
        detect_tokenize_text = self.processing_text.tokenize_detect(clean_text)

        

        for key in tokenize_text.keys():
            tokenize_text[key] = torch.tensor([tokenize_text[key]]).to(self.config.DEVICE)
            detect_tokenize_text[key] = torch.tensor(detect_tokenize_text[key]).to(self.config.DEVICE)
        
        self.model.eval()
        self.detect_model.eval()
        with torch.no_grad():
            rating_output = self.model(**tokenize_text)
            detect_output = self.detect_model(**detect_tokenize_text)
        
        prediction = self.get_final_prediction(rating_output.logits, detect_output.logits)
        return prediction

    def get_final_prediction(self, rating_logits, detect_logits):
        non_null = torch.max(rating_logits[:,1:], dim=-1)
        non_null_predictions = non_null.indices + 1
        rating_null_score = rating_logits[:,0] - non_null.values
        detect_null_score = detect_logits[:,0] - detect_logits[:,1]

        null_score = self.config.alpha * rating_null_score + self.config.beta * detect_null_score

        final_predictions = (null_score < self.config.threshold).type(torch.int)*non_null_predictions
        return final_predictions.cpu().numpy().tolist()

if __name__ == "__main__":
    import config
    from time import time
    RATING_ASPECTS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "di_chuyen", "mua_sam"]

    review_solver = ClassifyReviewSolver(config)

    review_sentence = "Nhà hàng ổn, đồ ăn ngon"
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

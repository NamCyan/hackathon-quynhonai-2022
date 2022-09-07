from utils import *
import torch

class TextProcessing:
    def __init__(self, tokenizer: object, max_length_rating: int, max_length_detect: int, padding: str = "longest") -> None:
        """
        :param sentence The review that we want to encode
        :param max_length: The maximum length of the input sequence
        :param batch_size: The number of samples to be processed in a single batch
        :param shuffle: Whether to shuffle the data or not, defaults to False (optional)
        """
        self.tokenizer = tokenizer
        self.max_length_rating = max_length_rating
        self.max_length_detect = max_length_detect
        self.padding = padding

    def clean_sentence(self, sentence: str) -> list:
    
        cleaned_sentence = fix_whitespace(remove_url(remove_emoji(sentence))).strip()
        return cleaned_sentence

    def tokenize_rate(self, sentence):
        return self.tokenizer(sentence, padding=self.padding, max_length= self.max_length_rating, truncation=True)

    def tokenize_detect(self, sentence):
        ASPECTS  =["Dịch vụ vui chơi giải trí",
                "Dịch vụ lưu trú",
                "Hệ thống nhà hàng phục vụ khách du lịch",
                "Dịch vụ ăn uống",
                "Dịch vụ di chuyển",
                "Dịch vụ mua sắm"]

        return self.tokenizer(ASPECTS, [sentence]*len(ASPECTS), padding=self.padding, max_length= self.max_length_detect, truncation="only_second")
    
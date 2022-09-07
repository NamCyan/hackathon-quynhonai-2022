import torch
import numpy as np
from constant import *
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaTokenizer

    
def load_sentence_pair(data_files, tokenizer, batch_size=32, max_length=256):
    data = load_dataset('csv', data_files= data_files)
    def tokenize(examples):
        return tokenizer(examples["aspect"], examples["review"], padding="max_length", truncation=True, max_length= max_length)

    input_data = data.map(tokenize, batched=True)
    print(input_data['train'][0]['review'])
    print(tokenizer.decode(input_data['train'][0]['input_ids']))
    return input_data, max(input_data['train']['label']) +1


if __name__ == "__main__":
    data_files = {'train': "/media/Z/namlh31/QuiNhon_hackathon/hackathon_data/train_processed.csv"}
    tokenizer = RobertaTokenizer.from_pretrained('/media/Z/namlh31/vi-roberta-part1')

    load_sentence_pair(data_files= data_files, tokenizer= tokenizer)


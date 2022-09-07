from json import detect_encoding
from transformers import (
    AutoConfig,
    AutoTokenizer,
    RobertaTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from model import RobertaMultiHeadClassifier, RobertaMixLayer, RobertaEnsembleLayer
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset, load_metric
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import logging
from tqdm.auto import tqdm
import torch
from eval_score import RA_score
import datasets

logger = logging.getLogger(__name__)

def get_model(model_path):
    config = AutoConfig.from_pretrained(
        model_path
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        model_path
    )
    tokenizer.do_lower_case = True

    if 'detection' in model_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
        )
    elif "mix" in model_path:
        model = RobertaMixLayer.from_pretrained(
            model_path,
            config=config,
        )
    elif "ensemble" in model_path:
        model = RobertaEnsembleLayer.from_pretrained(
            model_path,
            config=config,
        )
    else:
        model = RobertaMultiHeadClassifier.from_pretrained(
            model_path,
            config=config,
        )

    return tokenizer, model


def get_null_detect(model, dataset, batch_size):
    f1_metric = datasets.load_metric('f1')

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size= batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    null_score = []
    predictions = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = {k: v.cuda() for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        null_score.append(logits[:,0] - logits[:,1])
        predictions.append(torch.argmax(logits,dim=-1))
    
    null_score = torch.cat(null_score,dim=0).cpu()
    predictions = torch.cat(predictions,dim=0).cpu()

    targets = dataset['labels'].view(-1)
    print(f1_metric.compute(predictions= predictions, references= targets, average='macro'))


    null_score = torch.cat([null_score.unsqueeze(0), targets.unsqueeze(0)], dim=0)
    return null_score


def get_null_rating(model, dataset, batch_size):
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size= batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    null_score = []
    non_null_score = []
    predictions = []
    targets = []

    ra_metric = RA_score()
    f1_metric = load_metric("f1")
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = {k: v.cuda() for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits

        batch_len = len(batch['labels'])


        null_score.append((logits[:,0] - torch.max(logits[:,1:], dim=-1).values).reshape(6,batch_len).transpose(0,1))
        non_null_score.append((torch.argmax(logits[:,1:], dim=-1) + 1).reshape(6, batch_len).transpose(0,1))
        predictions.append(torch.argmax(logits, dim=-1).reshape(6, batch_len).transpose(0,1))

    null_score = torch.cat(null_score,dim=0).view(-1).cpu()
    non_null_score = torch.cat(non_null_score,dim=0).view(-1).cpu()
    predictions = torch.cat(predictions,dim=0).view(-1).cpu()
    targets = dataset['labels'].view(-1)
    
    print(ra_metric.compute(predictions= predictions, references= targets))
    print(f1_metric.compute(predictions= predictions, references= targets, average= "macro"))
    null_score = torch.cat([null_score.unsqueeze(0), non_null_score.unsqueeze(0), predictions.unsqueeze(0), targets.unsqueeze(0)], dim=0)
    return null_score

def main():
    RATING_MODEL_PATH = "../QuiNhonAI/vi-roberta-large-mix-PSUM-256/"
    DETECTION_MODEL_PATH = "../QuiNhonAI/vi-roberta-base-detection-weighted/"
    EVAL_DATA_PATH_JSON = "../QuiNhonAI/data/dev_processed.json"
    EVAL_DATA_PATH_CSV = "../QuiNhonAI/data/dev_processed.csv"
    CACHE_DIR = "../QuiNhonAI/cache"
    BATCH_SIZE = 1
    padding = "longest"
    max_seq_length_rating = 256
    max_seq_length_detect = 256

    rating_tokenizer, rating_model = get_model(RATING_MODEL_PATH)
    detect_tokenizer, detect_model = get_model(DETECTION_MODEL_PATH)

    rating_model.cuda()
    detect_model.cuda()

    rating_datasets = load_dataset(
        "json",
        data_files=EVAL_DATA_PATH_JSON,
        cache_dir=CACHE_DIR,
    )

    detect_datasets = load_dataset(
        "csv",
        data_files=EVAL_DATA_PATH_CSV,
        cache_dir=CACHE_DIR,
    )

    #######
    def preprocess_rating(examples):
        # Tokenize the texts
        result = rating_tokenizer(examples["review"], padding=padding, max_length=max_seq_length_rating, truncation=True)
        return result

    def preprocess_detection(examples):
        # Tokenize the texts
        result = detect_tokenizer(examples["aspect"], examples["review"], padding=padding, max_length=max_seq_length_detect, truncation="only_second")
        return result

    tokenized_rating_datasets = rating_datasets.map(
        preprocess_rating,
        batched=True,
        load_from_cache_file= False,
    ).remove_columns(["review", "id"]).rename_column("label", "labels")

    tokenized_detect_datasets = detect_datasets.map(
        preprocess_detection,
        batched=False,
        load_from_cache_file= False,
    ).map(lambda example: {"labels": 1 if example['label'] > 0 else 0}).remove_columns(['review', 'aspect', 'label'])

    tokenized_rating_datasets.set_format("torch")
    tokenized_detect_datasets.set_format("torch")

    detect_null_score = get_null_detect(detect_model, tokenized_detect_datasets['train'], BATCH_SIZE)
    print(detect_null_score)

    rating_null_score = get_null_rating(rating_model, tokenized_rating_datasets['train'], BATCH_SIZE)
    print(rating_null_score)
    
    torch.save(detect_null_score, "detect_null_score.torch")
    torch.save(rating_null_score, "rating_null_score.torch")

if __name__ == "__main__":
    main()
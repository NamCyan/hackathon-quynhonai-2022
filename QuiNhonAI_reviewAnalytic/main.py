from load_data import *
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from eval_score import *
import numpy as np
from transformers import Trainer, TrainingArguments
import argparse
from datasets import load_metric
import json


def get_args():
    parser = argparse.ArgumentParser(description='Spelling correction')
    # Arguments
    parser.add_argument('--seed', type=int, default=21, 
                        help='initial seed')

    parser.add_argument('--model_path', type=str, default="/media/Z/namlh31/vi-roberta-part1/", 
                        help='model name or path')
    parser.add_argument('--max_input_length', default=256, type=int,
                        help='maximum sequence length')
    parser.add_argument('--epochs', default=5, type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size')
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='gradient_accumulation_steps')
    parser.add_argument('--eval_steps', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('--save_steps', default=2000, type=int,
                        help='Number of training epochs')                   
    args = parser.parse_args()
    return args

args =get_args()
print(args)


data_files = {'train': "/media/Z/namlh31/QuiNhon_hackathon/hackathon_data/train_processed.csv", "valid": "/media/Z/namlh31/QuiNhon_hackathon/hackathon_data/dev_processed.csv"}
tokenizer = RobertaTokenizer.from_pretrained(args.model_path)


dataset, num_labels = load_sentence_pair(data_files= data_files, tokenizer= tokenizer, batch_size = args.batch_size, max_length= args.max_input_length)
print(dataset, num_labels)


model = RobertaForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels, from_flax= True)

metric = RA_score()
f1_metric = datasets.load_metric("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    ra_score = metric.compute(predictions=predictions, references=labels)
    f1_score = f1_metric.compute(predictions=predictions, references=labels, average= 'macro')
    return {"ra_score": ra_score, 'f1_score': f1_score}
# metric = load_metric("accuracy")
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir = "result",
    evaluation_strategy = "steps", #print evaluation after finishing an epoch
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    save_total_limit=1,
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    compute_metrics=compute_metrics
)

trainer.train()

test_eval = trainer.predict(dataset["valid"])
print(json.dumps(test_eval.metrics, indent= 4))

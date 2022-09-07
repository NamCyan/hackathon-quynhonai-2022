
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import itertools
import torch
from tqdm.auto import tqdm
import datasets
import numpy as np
from datasets import load_dataset
import json
from sklearn.metrics import confusion_matrix

from eval_score import RA_score
import transformers
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
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from copy import deepcopy
import torch

from model import RobertaMultiHeadClassifier


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    do_lower_case: bool = field(
        default=False, metadata={"help": "Lower case sentences"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    is_pair: bool = field(
        default=False, metadata={"help": "Sentence pair classification"}
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    model_type: Optional[str] = field(
        default=None,
    )


def evaluate(args, model, dataset, num_sample= 10, model_type= "pair", majority_vote= False):

    ra_metric = RA_score()
    f1_metric = datasets.load_metric("f1")

    results = {}

    batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size= batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)
    
    predictions = []
    p, g = [], []
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        batch = {k: v.to(args.device) for k, v in batch.items()}
        batch_len = len(batch['labels'])

        if num_sample is not None:
            batch['num_sample'] = num_sample
            batch['majority_vote'] = majority_vote
        with torch.no_grad():
            if num_sample is not None:
                outputs = model.MC_predict(**batch)
            else:
                outputs = model(**batch)

        logits = outputs.logits
        
        predictions.append(torch.argmax(logits, dim=-1).reshape(6, batch_len).transpose(0,1))
        if majority_vote:
            predictions = logits
    
    predictions = torch.cat(predictions,dim=0).view(-1).cpu()
    targets = dataset['labels'].view(-1)
        
    ra_score = ra_metric.compute(predictions= predictions, references= targets)
    f1_score = f1_metric.compute(predictions= predictions, references= targets, average="macro")
    
    
    results["ra_score"] = list(ra_score.values())[0]
    results["f1_score"] = list(f1_score.values())[0]
    results["confusion_matrix"] = confusion_matrix(targets, predictions)
    return results



parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

if model_args.model_type == "attention":
    from model import RobertaAspectEmbedding as model_class
elif model_args.model_type == "pair":
    from transformers import AutoModelForSequenceClassification as model_class
elif model_args.model_type == "mixlayer":
    from model import RobertaMixLayer as model_class
else:
    from model import RobertaMultiHeadClassifier as model_class


config = AutoConfig.from_pretrained(
            training_args.output_dir
)
if 'phobert' in model_args.model_name_or_path or 'xlmr' in model_args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.output_dir
    )
else:
    tokenizer = RobertaTokenizer.from_pretrained(
        training_args.output_dir
    )
tokenizer.do_lower_case = data_args.do_lower_case


model = model_class.from_pretrained(
    training_args.output_dir,
    config=config,
)
model.to(training_args.device)


data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
model.to(training_args.device)

# Padding strategy
if data_args.pad_to_max_length:
    padding = "max_length"
else:
    # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    padding = False

# Some models have set the order of the labels to use, so let's make sure we do use it.

if data_args.max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    # Tokenize the texts
    if data_args.is_pair:
        result = tokenizer(examples["aspect"], examples["review"], padding=padding, max_length=max_seq_length, truncation="only_second")
    else:
        result = tokenizer(examples["review"], padding=padding, max_length=max_seq_length, truncation=True)
    return result

with training_args.main_process_first(desc="dataset map pre-processing"):
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

tokenized_datasets = tokenized_datasets.remove_columns(["review", "id"]) if not model_args.model_type == "pair" else tokenized_datasets.remove_columns(['review', 'aspect']) 
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

print(tokenizer.decode(tokenized_datasets['train'][0]['input_ids']))


result = evaluate(training_args, model, tokenized_datasets['validation'], model_type= model_args.model_type, num_sample=None, majority_vote= None)
print(result)

print(result['confusion_matrix'])
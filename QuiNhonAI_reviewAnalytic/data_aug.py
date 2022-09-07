import json
from datasets import load_dataset
from transformers import set_seed
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import random
import jsonlines
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# raw_datasets = load_dataset(
#             "json",
#             data_files="../QuiNhonAI/data/train_processed.json",
# )


# set_seed(21)

# augs = {}

# # augs["random_insert"] = naw.ContextualWordEmbsAug(model_path="../QuiNhonAI/review-roberta-large/", 
# #                         device="cuda", action="insert", aug_max=1)
# # augs["random_swap"] = naw.RandomWordAug(action="swap")
# # augs["random_delete"] = naw.RandomWordAug()

# augs["random_substitute"] = naw.ContextualWordEmbsAug(model_path = "../QuiNhonAI/review-roberta-large/", 
#                         device='cuda', action="substitute")

# # augs["bt_vi_en"] = naw.BackTranslationAug(
# #     from_model_name='NlpHUST/t5-vi-en-base', 
# #     to_model_name='NlpHUST/t5-en-vi-base',
# #     max_length=512, device= 'cuda'
# # )

# # choices = ["bt_vi_en", "random_substitute"]
# id = 0
# augment_data = []
# for data in raw_datasets['train']:      
# #     # augment_data.append(data)

# #     # randnum = random.randint(1,2)
# #     # if randnum == 1:
# #     #     aug_method = random.choice(choices)

#     text = data['review']
#     aug_text = augs['random_substitute'].augment(text)[0]
#     augment_data.append({'review': aug_text, 'id': "aug_" + str(len(augment_data)), 'label': data['label']}) 
#     # else:
#     #     for aug_method in choices:
#     #         aug_text = augs[aug_method].augment(text)[0]
#     #         augment_data.append({'review': aug_text, 'id': "aug_" + str(len(augment_data)), 'label': data['label']})
            

# with jsonlines.open('../QuiNhonAI/data/train_augment_subtitute.json', 'w') as writer:
#     writer.write_all(augment_data)

all_data = []
with jsonlines.open("../QuiNhonAI/data/train_augment_subtitute.json") as reader:
    for i in reader:
        all_data.append(i)

with jsonlines.open("../QuiNhonAI/data/train_processed.json") as reader:
    for i in reader:
        i['id'] = "train_" + str(i['id'])
        all_data.append(i)
with jsonlines.open('../QuiNhonAI/data/train_augment_subtitute_.json', 'w') as writer:
    writer.write_all(all_data)
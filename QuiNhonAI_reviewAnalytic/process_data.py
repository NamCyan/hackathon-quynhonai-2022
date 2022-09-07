import torch
import numpy as np
from constant import *
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from utils import remove_emoji, remove_url, fix_whitespace
from sklearn.model_selection import train_test_split
import jsonlines

# df = pd.read_csv("../data/data_final_problem2.csv")
# rng = np.random.RandomState()

# train = df.sample(frac=0.9, random_state=rng)
# dev = df.loc[~df.index.isin(train.index)]

# raw_datas = [df, train, dev]
# names = ["full_processed",'train_processed','dev_processed']



# for i, df in enumerate(raw_datas):
#     processed_data = []
#     for dp in df.to_dict('records'):
#         for asp in map_label2fulllabel.keys():
#             processed_dp = {}
#             processed_dp['review'] = fix_whitespace(remove_url(remove_emoji(dp['Review'])))
#             processed_dp['aspect'] = map_label2fulllabel[asp]
#             processed_dp['label'] = dp[asp]

#             processed_data.append(processed_dp)

#     datacsv = pd.DataFrame.from_records(processed_data)
#     print(len(datacsv))
#     print(datacsv.head())
#     datacsv.to_csv("../data/{}.csv".format(names[i]), index= False)


# import vncorenlp

# # Automatically download VnCoreNLP components from the original repository
# # and save them in some local machine folder
# # py_vncorenlp.download_model(save_dir='../vncorenlp')

# # Load VnCoreNLP
# rdrsegmenter = vncorenlp.VnCoreNLP('../vncorenlp/VnCoreNLP-1.1.1.jar', annotators="wseg")

# train = pd.read_csv("/media/Z/namlh31/QuiNhon_hackathon/hackathon_data/train_processed.csv")
# valid = pd.read_csv("/media/Z/namlh31/QuiNhon_hackathon/hackathon_data/dev_processed.csv")
# train_data = []

# index= 0
# for i in range(0, len(train), 6):
#     dp = train[i:i+6]
#     review= list(dp['review'])[0]
#     label= list(dp['label'])

#     output = []
#     for sentence in rdrsegmenter.annotate(review)['sentences']:
#         output += [x['form'] for x in sentence]

#     train_data.append({'id': index, 'review': " ".join(output), 'label': label})
#     index +=1

# print(len(train_data))
# with jsonlines.open("/media/Z/namlh31/QuiNhon_hackathon/hackathon_data/train_processed_wseg.json", mode='w') as writer:
#     writer.write_all(train_data)

# valid_data = []

# index= 0
# for i in range(0, len(valid), 6):
#     dp = valid[i:i+6]

#     review= list(dp['review'])[0]
#     label= list(dp['label'])

#     output = []
#     for sentence in rdrsegmenter.annotate(review)['sentences']:
#         output += [x['form'] for x in sentence]

#     valid_data.append({'id': index, 'review': " ".join(output), 'label': label})
#     index +=1

# print(len(valid_data))
# with jsonlines.open("/media/Z/namlh31/QuiNhon_hackathon/hackathon_data/dev_processed_wseg.json", mode='w') as writer:
#     writer.write_all(valid_data)


#split by aspe
data = [[] for i in range(6)]
with jsonlines.open("../QuiNhonAI/data/dev_processed.json", "r") as reader:
    for dp in reader:
        for i, label in enumerate(dp['label']):
            data[i].append({'id': dp['id'],'review': dp['review'], 'label': label})
for i, dt in enumerate(data):
    with jsonlines.open("../QuiNhonAI/data/dev_split_{}.json".format(str(i)), mode='w') as writer:
        writer.write_all(dt)
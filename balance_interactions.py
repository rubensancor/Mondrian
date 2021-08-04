import os
import ast
import json
import pickle
import argparse
import datetime
import random
from tqdm import tqdm
from collections import Counter
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Data file')
args = parser.parse_args()

data = pd.read_csv(args.data)
heads = data['head'].tolist()
users = []
types = []
legit_list = []
first_time = last_time = None

for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    if row['label_1'] == 1 and first_time is None:
        first_time = last_time = row['timestamp']
    if row['label_1'] == 1 and row['timestamp'] > last_time:
        last_time = row['timestamp']
    if row['label_1'] == 0:
        legit_list.append(row['head'])
    if row['head'] not in users:
        users.append(row['head'])
        types.append(row['label_1'])
    else:
        continue


print(len(set(users)))
print(Counter(types))

random.shuffle(users)
legit_users70 = int((Counter(types)[1] * 70) / 30)
legit_ids = list(set(legit_list))[:legit_users70]
legit_users = legit_ids

legit_users.insert(0, str(str(first_time) + ' ' + str(last_time)))
with open(args.data.split('_')[0] + '_legit_users.txt', 'w') as f:
    for item in legit_users:
        f.write("%s\n" % item)

df = data[0:0]


print('### CREATING NEW DF ###')
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    if row['label_1'] == 1 or row['head'] in legit_ids:
        entry = data.loc[[index]]
        df = pd.concat([df,entry], ignore_index=True)
    
print('### SAVING NEW DATA ###')
# users = []
# types = []
# for index, row in tqdm(data.iterrows(), total=data.shape[0]):
#     if row['head'] not in users:
#         users.append(row['head'])
#         types.append(row['label_1'])
#     else:
#         continue


df = df.sort_values('timestamp')
df.to_csv(args.data.split('.')[0] + '_balanced.csv', index=False)

    




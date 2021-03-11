# user_id,item_id,timestamp,state_label,comma_separated_list_of_features

from collections import Counter
import argparse
import pandas as pd
from tqdm import tqdm

item_conversion = {
    'mention-hashtag': 0, 
    'mention': 1, 
    'tweet': 2, 
    'mention-hashtag-url': 3, 
    'mention-url': 4, 
    'reply': 5, 
    'rt-hashtag': 6, 
    'reply-hashtag': 7, 
    'rt': 8, 
    'reply-url': 9, 
    'reply-hashtag-url': 10, 
    'rt-hashtag-url': 11, 
    'rt-url': 12}

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data', required=True, help='Data file')
args = parser.parse_args()

data = pd.read_csv(args.data)

d = {}
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    d[index] = {
        'user_id': row['head'],
        'item_id': item_conversion[row['interaction']],
        'timestamp': row['timestamp'],
        'state_label': 0,
        'comma_separated_list_of_features': 0
    }

df = pd.DataFrame.from_dict(d, orient='index')
df = df.sort_values('timestamp')
df.to_csv(args.data + '_jodie.csv', index=False)


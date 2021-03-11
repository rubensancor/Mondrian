import os
import ast
import json
import pickle
import argparse
import datetime
from tqdm import tqdm
import pandas as pd

def check_user(user, checklist):
    if user in checklist:
        return '1'
    else:
        return '0'

# Converts tweets from the dataset in interactions with this structure
#(head, tail, interaction, timestamp, label_1, label_2)
parser = argparse.ArgumentParser()
parser.add_argument('--folder', required=True, help='Folder path')
args = parser.parse_args()

dict_inter = {}
t = 0
malicious_users = []
first_ts = None
last_ts = None
cnt = 0
for f in sorted(os.listdir(args.folder)):
    file_type = f.split('.')[-1]

    if file_type == 'csv':
        data = pd.read_csv(args.folder + f)


        # Define the malicious users
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            ts = datetime.datetime.timestamp(datetime.datetime.strptime(row['tweet_time'], "%Y-%m-%d %H:%M"))
            if first_ts is None:
                first_ts = ts
                last_ts = ts
            if ts < first_ts:
                first_ts = ts
            elif ts > last_ts:
                last_ts = ts
            malicious_users.append(row['userid'])
            cnt += 1
        first_ts -= 172800
        last_ts += 172800

        malicious_users = list(set(malicious_users))
        
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            head = row['userid']
            date = datetime.datetime.strptime(row['tweet_time'], "%Y-%m-%d %H:%M")
            timestamp = datetime.datetime.timestamp(date)

            label_1 = check_user(head, malicious_users)

            if isinstance(row['in_reply_to_userid'], str):
                replies = row['in_reply_to_userid'].strip('][').split(', ')
                
                for reply in replies:
                    tail = reply.replace("'",'')
                    if tail == '0' or tail == '':
                        continue

                    label_2 = check_user(tail, malicious_users)
                    
                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': 'reply',
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2
                    }
                    t += 1

            if isinstance(row['retweet_userid'], str):
                retweets = row['retweet_userid'].strip('][').split(', ')

                for rt in retweets:
                    tail = rt.replace("'",'')
                    if tail == '0' or tail == '':
                        continue

                    label_2 = check_user(tail, malicious_users)
                    
                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': 'rt',
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2
                    }
                    t += 1

            if isinstance(row['user_mentions'], str) and not isinstance(row['retweet_userid'], str):
                if row['user_mentions'] == '[]':
                    continue
                mentions = ast.literal_eval(row['user_mentions'])
                for mention in mentions:
                    tail = mention
                    if tail == '0' or tail == 0 or tail == '':
                        continue

                    label_2 = check_user(tail, malicious_users)

                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': 'mention',
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2
                    }
                    t += 1

            if isinstance(row['hashtags'], str) and not row['hashtags'] == '[]':
                hashtags = ast.literal_eval(row['hashtags'])
                for hashtag in hashtags:
                    tail = hashtag.lower()
                    if tail == '':
                        continue

                    label_2 = '2'
                        
                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': 'hashtag',
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2
                    }
                    t += 1

            if isinstance(row['urls'], str) and not row['urls'] == '[]':
                urls = ast.literal_eval(row['urls'])
                for url in urls:
                    tail = url
                    if tail == '':
                        continue

                    label_2 = '3'
                        
                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': 'url',
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2
                    }
                    t += 1
                    

    elif file_type == 'pickle':
        with open(args.folder + f, 'rb') as pickleFile:
            data = pickle.load(pickleFile)
        for tweet in tqdm(data):
            if len(tweet) < 20:
                continue
            tweet = json.loads(tweet[0:-2])

            if tweet['id'] is not None:
                head = tweet['id']
            else:
                continue

            label_1 = check_user(head, malicious_users)
            
            date = datetime.datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S %z %Y")
            timestamp = datetime.datetime.timestamp(date)

            if timestamp < first_ts or timestamp > last_ts:
                continue


            if tweet['in_reply_to_user_id'] is not None:
                tail = tweet['in_reply_to_user_id']
                
                label_2 = check_user(tail, malicious_users)

                dict_inter[t] = {
                    'head': head,
                    'tail': tail,
                    'interaction': 'reply',
                    'timestamp': timestamp,
                    'label_1': label_1,
                    'label_2': label_2
                }
                t += 1

            if tweet['entities']['user_mentions'] is not None and len(tweet['entities']['user_mentions']) > 0:
                for user in tweet['entities']['user_mentions']:
                    tail = user['id']

                    label_2 = check_user(tail, malicious_users)
                    
                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': 'mention',
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2
                    }
                    t += 1

            if len(tweet['entities']['hashtags']) > 0:
                for hashtag in tweet['entities']['hashtags']:
                    tail = hashtag['text'].lower()

                    label_2 = '2'
                        
                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': 'hashtag',
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2
                    }
                    t += 1

            if len(tweet['entities']['urls']) > 0:
                for hashtag in tweet['entities']['urls']:
                    tail = hashtag['url']

                    label_2 = '3'
                        
                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': 'url',
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2
                    }
                    t += 1

    
df = pd.DataFrame.from_dict(dict_inter, orient='index')
df = df.sort_values('timestamp')
df.to_csv(args.folder.split('/')[1] + '_interactions.csv', index=False)



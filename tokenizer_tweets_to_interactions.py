import os
import ast
import json
import pickle
import argparse
import datetime
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer

def check_user(user, checklist):
    if user in checklist:
        return '1'
    else:
        return '0'
    
def create_hashtag_tokens(hashtags):
    try:
        if isinstance(hashtags, str):
            hashtags = hashtags.strip("'[]").split(',')
        tokens = tokenizer(hashtags, is_split_into_words=True)['input_ids']
    except:
        tokens = [0]
        
    return tokens

# Converts tweets from the dataset in interactions with this structure
#(head, tail, interaction, timestamp, label_1, label_2, hashtag_tokens_separated_by_comma)
parser = argparse.ArgumentParser()
parser.add_argument('--folder', required=True, help='Folder path')
parser.add_argument('--users', action='store_true', help='If they are user interactions.')
parser.add_argument('--disable_twitter', action='store_true', help='Disable tweets without interaction.')
parser.add_argument('--disable_rts', action='store_true', help='Disable rts.')
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_token_size = 1

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
            malicious_users.append(row['userid'])
            ts = None
            try:
                ts = datetime.datetime.timestamp(datetime.datetime.strptime(row['tweet_time'], "%Y-%m-%d %H:%M"))
            except ValueError:
                continue
            if first_ts is None:
                first_ts = ts
                last_ts = ts
            if ts < first_ts:
                first_ts = ts
            elif ts > last_ts:
                last_ts = ts
            cnt += 1
        first_ts -= 172800
        last_ts += 172800

        malicious_users = list(set(malicious_users))
        
        
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            head = row['userid']
            try:
                date = datetime.datetime.strptime(row['tweet_time'], "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            
            timestamp = datetime.datetime.timestamp(date)

            label_1 = check_user(head, malicious_users)

            if isinstance(row['in_reply_to_userid'], str):
                replies = row['in_reply_to_userid'].strip('][').split(', ')
                
                for reply in replies:
                    tail = reply.replace("'",'')
                    if tail == '0' or tail == '':
                        continue

                    label_2 = check_user(tail, malicious_users)

                    interaction = 'reply'
                    if isinstance(row['hashtags'], str) and not row['hashtags'] == '[]':
                        interaction = 'reply-hashtag'
                        if isinstance(row['urls'], str) and not row['urls'] == '[]':
                            interaction = 'reply-hashtag-url'
                    elif isinstance(row['urls'], str) and not row['urls'] == '[]':
                        interaction = 'reply-url'
                    
                    
                    tokens = create_hashtag_tokens(row['hashtags'])
                    
                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': interaction,
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2,
                        'features': tokens
                    }
                    t += 1

            elif isinstance(row['retweet_userid'], str):
                retweets = row['retweet_userid'].strip('][').split(', ')

                for rt in retweets:
                    tail = rt.replace("'",'')
                    if tail == '0' or tail == '':
                        continue

                    label_2 = check_user(tail, malicious_users)

                    interaction = 'rt'
                    if isinstance(row['hashtags'], str) and not row['hashtags'] == '[]':
                        interaction = 'rt-hashtag'
                        if isinstance(row['urls'], str) and not row['urls'] == '[]':
                            interaction = 'rt-hashtag-url'
                    elif isinstance(row['urls'], str) and not row['urls'] == '[]':
                        interaction = 'rt-url'
                        
                    tokens = create_hashtag_tokens(row['hashtags'])
                    
                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': interaction,
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2,
                        'features': tokens
                    }
                    t += 1

            elif isinstance(row['user_mentions'], str) and not isinstance(row['retweet_userid'], str):
                if row['user_mentions'] == '[]':
                    continue
                mentions = ast.literal_eval(row['user_mentions'])
                for mention in mentions:
                    tail = mention
                    if tail == '0' or tail == 0 or tail == '':
                        continue

                    label_2 = check_user(tail, malicious_users)

                    interaction = 'mention'
                    if isinstance(row['hashtags'], str) and not row['hashtags'] == '[]':
                        interaction = 'mention-hashtag'
                        if isinstance(row['urls'], str) and not row['urls'] == '[]':
                            interaction = 'mention-hashtag-url'
                    elif isinstance(row['urls'], str) and not row['urls'] == '[]':
                        interaction = 'mention-url'
                        
                    tokens = create_hashtag_tokens(row['hashtags'])

                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': interaction,
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2,
                        'features': tokens
                    }
                    t += 1

            else:
                if not args.disable_twitter:
                    tokens = create_hashtag_tokens(row['hashtags'])
                    
                    dict_inter[t] = {
                            'head': head,
                            'tail': 'Twitter',
                            'interaction': 'tweet',
                            'timestamp': timestamp,
                            'label_1': label_1,
                            'label_2': 3,
                            'features': tokens
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
                head = tweet['user']['id_str']
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

                interaction = 'reply'
                if len(tweet['entities']['hashtags']) > 0:
                    interaction = 'reply-hashtag'
                    if len(tweet['entities']['urls']) > 0:
                        interaction = 'reply-hashtag-url'
                elif len(tweet['entities']['urls']) > 0:
                    interaction = 'reply-url'
                
                
                hashtags = []
                for hashtag in tweet['entities']['hashtags']:
                    hashtags.append(hashtag['text'])
                
                tokens = create_hashtag_tokens(hashtags)
                
                dict_inter[t] = {
                    'head': head,
                    'tail': tail,
                    'interaction': interaction,
                    'timestamp': timestamp,
                    'label_1': label_1,
                    'label_2': label_2,
                    'features': tokens
                }
                t += 1

            elif tweet['entities']['user_mentions'] is not None and len(tweet['entities']['user_mentions']) > 0:
                for user in tweet['entities']['user_mentions']:
                    tail = user['id']

                    label_2 = check_user(tail, malicious_users)
                    
                    interaction = 'mention'
                    if len(tweet['entities']['hashtags']) > 0:
                        interaction = 'mention-hashtag'
                        if len(tweet['entities']['urls']) > 0:
                            interaction = 'mention-hashtag-url'
                    elif len(tweet['entities']['urls']) > 0:
                        interaction = 'mention-url'
                       
                    hashtags = [] 
                    for hashtag in tweet['entities']['hashtags']:
                        hashtags.append(hashtag['text'])
                
                    tokens = create_hashtag_tokens(hashtags)

                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': interaction,
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2,
                        'features': tokens
                    }
                    t += 1
                    
            elif tweet['is_quote_status']:
                try:
                    tail = tweet['quoted_status']['user']['id']
                    if tail == 0 or tail == None:
                        continue
                    
                    label_2 = check_user(tail, malicious_users)

                    interaction = 'rt'
                    if len(tweet['entities']['hashtags']) != 0 or len(tweet['quoted_status']['entities']['hashtags']) != 0:
                        interaction = 'rt-hashtag'
                        if len(tweet['entities']['urls']) != 0 or len(tweet['quoted_status']['entities']['urls']) != 0:
                            interaction = 'rt-hashtag-url'
                    elif len(tweet['entities']['urls']) != 0 or len(tweet['quoted_status']['entities']['urls']) != 0:
                        interaction = 'rt-url'
                        
                    hashtags = [] 
                    for hashtag in tweet['entities']['hashtags']:
                        hashtags.append(hashtag['text'])
                    for hashtag in tweet['quoted_status']['entities']['hashtags']:
                        hashtags.append(hashtag['text'])
                    
                    # for rt in retweets:
                    #     tail = rt.replace("'",'')
                    #     if tail == '0' or tail == '':
                    #         continue

    
                    tokens = create_hashtag_tokens(hashtags)
                    
                    dict_inter[t] = {
                        'head': head,
                        'tail': tail,
                        'interaction': interaction,
                        'timestamp': timestamp,
                        'label_1': label_1,
                        'label_2': label_2,
                        'features': tokens
                    }
                    t += 1
                except:
                    continue
            else:
                if not args.disable_twitter:
                    hashtags = []
                    for hashtag in tweet['entities']['hashtags']:
                        hashtags.append(hashtag['text'])
                    
                    tokens = create_hashtag_tokens(hashtags)
                    
                    dict_inter[t] = {
                            'head': head,
                            'tail': 'Twitter',
                            'interaction': 'tweet',
                            'timestamp': timestamp,
                            'label_1': label_1,
                            'label_2': 3,
                            'features': tokens
                        }
                    t += 1


for i in dict_inter:
    if len(dict_inter[i]['features']) > max_token_size:
        max_token_size = len(dict_inter[i]['features'])

for i in dict_inter:
    if len(dict_inter[i]['features']) < max_token_size:
        padding = max_token_size - len(dict_inter[i]['features'])
        tmp = [0] * padding
        dict_inter[i]['features'].extend(tmp)


df = pd.DataFrame.from_dict(dict_inter, orient='index')
df = df.sort_values('timestamp')

write_name = 'interactions.csv'
if args.disable_twitter:
    write_name = 'NO_TWITTER_' + write_name
if args.users:
    df.to_csv(args.folder.split('/')[-2] + '_users_' + write_name, index=False)
else:
    df.to_csv(args.folder.split('/')[-2] + write_name, index=False)



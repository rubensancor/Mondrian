import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import torch
import random
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import scale


def load_data(args, time_scaling=True):
    
    head_list = []
    tail_list = []
    action_list = []
    timestamp_list = []
    head_labels = []
    tail_labels = []
    start_timestamp = None
    feature_list = []
    
    df = pd.read_csv(args.data)
    # if args.split != 0:
    #     rows = len(df.index)
    #     limit = int(rows * (0.15 + (0.05 * (args.split-1)))) # 10%+5%*split is training set + 5% for test set
    #     df = df.head(limit)

    print('*** Loading dataset')
    head_list = df['head'].tolist()
    tail_list = df['tail'].tolist()
    action_list = df['interaction'].tolist()
    head_labels = df['label_1'].tolist()
    tail_labels = df['label_2'].tolist()
    feature_list = [[0] * 1] * len(head_list)
    tmp_list = df['timestamp'].tolist()
    start_timestamp = tmp_list[0]
    timestamp_list = [timestamp - start_timestamp for timestamp in tmp_list]
    
    head_list = np.array(head_list)
    tail_list = np.array(tail_list)
    timestamp_list = np.array(timestamp_list)

    
    print('Formating actions to ids')
    nodeid = 0
    action2id = {}
    action_timediference_list = []
    action_current_timestamp = defaultdict(float)
    for cnt, action in enumerate(action_list):
        if action not in action2id:
            action2id[action] = nodeid
            nodeid += 1
        timestamp = timestamp_list[cnt]
        action_timediference_list.append(timestamp - action_current_timestamp[action])
        action_current_timestamp[action] = timestamp
    num_actions = len(action2id)
    action_list_id = [action2id[action] for action in action_list]

    print('Formating users to ids')
    nodeid = 0
    user2id = {}
    user_timediference_list = []
    user_current_timestamp = defaultdict(float)
    user_previous_actionid_list = []
    user_lastest_actionid = defaultdict(lambda: num_actions)
    for cnt, user in enumerate(head_list):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_list[cnt]
        user_timediference_list.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_actionid_list.append(user_lastest_actionid[cnt])
        user_lastest_actionid[user] = action2id[action_list[cnt]]
    
    num_users = len(user2id)
    user_list_id = [user2id[user] for user in head_list]

    if time_scaling:
        print('Scaling timestamps')
        user_timediference_list = scale(np.array(user_timediference_list) + 1)
        action_timediference_list = scale(np.array(action_timediference_list) + 1)
    
    print('*** Loading completed ***\n\n')
    

    return (user2id, user_list_id, user_timediference_list, user_previous_actionid_list,
            action2id, action_list_id, action_timediference_list, 
            timestamp_list,
            feature_list, 
            head_labels, 
            tail_labels)

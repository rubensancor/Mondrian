import ast
import random
from collections import defaultdict, Counter
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import scale


def load_data(args, time_scaling=True, tail_as_feature=False):
    
    head_list = []
    tail_list = []
    action_list = []
    timestamp_list = []
    head_labels = []
    tail_labels = []
    start_timestamp = None
    feature_list = []
    reduced_head_list = []
    
    df = pd.read_csv(args.data)

    # is_NaN = df.isnull()
    # row_has_NaN = is_NaN.any(axis=1)
    # rows_with_NaN = df[row_has_NaN]
    
    # for ind, i in enumerate(row_has_NaN):
    #     if i == True:
    #         print(ind)
    # exit()
    print('*** Loading dataset')
    head_list = df['head'].tolist()
    tail_list = df['tail'].tolist()
    action_list = df['interaction'].tolist()
    head_labels = df['label_1'].tolist()
    tail_labels = df['label_2'].tolist()
    tweet_list = df['tweet'].tolist()
    feature_list = df['features'].tolist()
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
    
    print('Formating feature')
    for index, i in enumerate(feature_list):
        l = ast.literal_eval(i)
        if tail_as_feature:
            l.insert(0, user_list_id[index])
        feature_list[index] = l

    if time_scaling:
        print('Scaling timestamps')
        user_timediference_list = scale(np.array(user_timediference_list) + 1)
        action_timediference_list = scale(np.array(action_timediference_list) + 1)
        
    print('Formating reduced head list')
    for i in range(num_users):
        reduced_head_list.append(head_labels[user_list_id.index(i)])
    
    
    print('*** Loading completed ***\n\n')
    

    return (user2id, user_list_id, user_timediference_list, user_previous_actionid_list,
            action2id, action_list_id, action_timediference_list, 
            timestamp_list,
            feature_list, 
            head_labels, 
            tail_labels,
            reduced_head_list,
            tweet_list)

def get_edited_users(args):
    if 'train_split' in vars(args):
        split = args.train_split
    else:
        split = args.data_split
        
    if '/' in args.data:
        tmp_data = args.data.split('/')[-1]
        filename = "./" + "saved_models/%s/checkpoint.%s.ep%d.tp%.2f.pth.tar" % (tmp_data, args.model, args.epochs, split)
    else:
        filename = "./" + "saved_models/%s/checkpoint.%s.ep%d.tp%.2f.pth.tar" % (args.data, args.model, args.epochs, split)
    checkpoint = torch.load(filename)
    try:
        edited_users = checkpoint['edited_users'] 
    except KeyError:
        print('There are no edited users saved in this checkpoint')
        raise
    return edited_users
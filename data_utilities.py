import os
import pandas as pd
import numpy as np
from collections import Counter

def rename(path):
    files = os.listdir(path)

    for file in files:
        os.rename(os.path.join(path, file), os.path.join(path, file.replace('\^J', '')))
        
def dataset_reducer(path, N):
    df = pd.read_csv(path)
    
    head_list = df['head'].tolist()
    head_set = set(head_list)
    head_labels = []
    for i in head_set:
        df_tmp = df.loc[df['head'] == i]
        head_labels.append(df_tmp.iloc[0]['label_1'])
    df_tmp = pd.DataFrame({'head': list(head_set),
                        'head_labels': head_labels})
    
    df_tmp = df_tmp.groupby('head_labels', group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(df_tmp))))).sample(frac=1).reset_index(drop=True)
    
    df_reduced = df.loc[df['head'].isin(df_tmp['head'].tolist())]
    
    df_reduced.to_csv(path.split('.')[0] + '_short.csv', index=False)
    
def dataset_stats(path):
    df = pd.read_csv(path)
    
    df_malicious = df[df.label_1 == 1]
    print('--- Malicious statistics ---')
    print('Number of users -----> %i' % len(set(df_malicious['head'].tolist())))
    print('Mean tweets per user -----> %f ' % df_malicious.groupby(['head']).size().mean())
    print(df_malicious.groupby(['interaction']).size() / len(set(df_malicious['head'].tolist())))
    
    df_legit = df[df.label_1 == 0]
    print('--- Legit statistics ---')
    print('Number of users -----> %i' % len(set(df_legit['head'].tolist())))
    print('Mean tweets per user -----> %f ' % df_legit.groupby(['head']).size().mean())
    print(df_legit.groupby(['interaction']).size() / len(set(df_legit['head'].tolist())))
    
def dataset_size(path):
    # TODO: Tiempo que comprende el dataset
    return

    
    

    
if __name__ == "__main__":
    # print('***** CHINA NO TWITTER *****')
    # dataset_stats('china-2_users_NO_TWITTER_interactions.csv')
    
    # print('***** SPAIN *****')
    # dataset_stats('spain_users_interactions_hashtagtoken.csv')
    
    # print('***** SPAIN NO TWITTER *****')
    # dataset_stats('spain_users_NO_TWITTER_interactions.csv')
    
    # dataset_stats('spain_users_NO_RT_NO_TWITTER_interactions.csv')
    dataset_stats('iran-1_users_NO_TWITTER_interactions.csv')
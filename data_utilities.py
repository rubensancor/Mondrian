import os
import pandas as pd
import numpy as np
from collections import Counter

def rename(path):
    files = os.listdir(path)

    for file in files:
        os.rename(os.path.join(path, file), os.path.join(path, file.replace('\^J', '').replace('\n', '')))
        
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
    print(path)
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

    
# def prepare_test_splits():
    
    

    
if __name__ == "__main__":    
    rename('data/userdata/honduras')
    
    # russia = pd.read_csv('russia_users_NO_TWITTER_normalized.csv')
    # china = pd.read_csv('china-2_users_NO_TWITTER_normalized.csv')
    # iran = pd.read_csv('iran-1_users_NO_TWITTER_normalized.csv')

    # out = russia.append(iran)
    # out.to_csv('russia_iran.csv', index=False)
import os
import argparse
import numpy as np
from data_loader import *
from sentence_transformers import SentenceTransformer


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', required=True, help='Data file')
args = parser.parse_args()

# Set the name of the data file 
args.dataname = args.data.split('.')[0]

if '/' in args.dataname:
    args.dataname = args.dataname.split('/')[-1]

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

df = pd.read_csv(args.data)
tweet_list = df['tweet'].tolist()

tweet_embedding_list = []
for j in tqdm(range(len(tweet_list))):
    tweet_embedding_list.append(model.encode(tweet_list[j]))
    
directory = os.path.join('./', 'embeddings/tweets/%s' % args.dataname)
        
np.savetxt(directory + '.txt', tweet_embedding_list)
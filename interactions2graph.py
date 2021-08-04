import argparse
import pandas as pd
import networkx as nx
from tqdm import tqdm, trange, tqdm_notebook, tnrange

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Data file')
args = parser.parse_args()


G = nx.MultiDiGraph()


with open(args.data, 'r') as f:
    
    df = pd.read_csv(f)
    
    # Retrieve nodes from dataframe and add them to the graph
    nodes = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['head'] not in G.nodes:
            G.add_node(row['head'], type=row['label_1'])
        # if row['tail'] not in G.nodes:
        #     G.add_node(row['tail'], type=row['label_2'])
        
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row['tail'] in G.nodes:
            G.add_edge(row['head'], row['tail'], interaction=row['interaction'],
                    start=float(row['timestamp']),
                    end=float(row['timestamp']))
        
    nx.write_gexf(G, args.data.split('.')[0]+'_NOTAIL.gexf')

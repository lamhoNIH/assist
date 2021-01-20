from node2vec import Node2Vec
import pandas as pd
import numpy as np
import os
from sys import platform 

def network_embedding(graph, walk_length, num_walks, window, output_dir = None, name_spec = ''):
    '''
    graph: networkx G for a network
    walk_length, num_walks, window: parameters for embedding
    name_spec: any additional info as str to add for saving the embedding df
    '''
    if platform == 'win32': # windows doesn't support multiple workers
        workers = 1
    else: # I don't know if this will work on MAC since I don't have one so George will test this to see if the code breaks
        workers = 1 # tried 4 and 2 on mac and both failed
    node2vec = Node2Vec(graph, dimensions=64, walk_length=walk_length, num_walks=num_walks, workers = workers)
    model = node2vec.fit(window = window, min_count=1)
    emb_df = pd.DataFrame(np.asarray(model.wv.vectors), index = graph.nodes)
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        emb_df.to_csv(f'{output_dir}/embedded_len{walk_length}_walk{num_walks}_{name_spec}.csv')
        print('embedding data saved')
    return emb_df


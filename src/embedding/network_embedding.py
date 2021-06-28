from node2vec import Node2Vec
import pandas as pd
import numpy as np
import os
from sys import platform 
import csrgraph as cg
import nodevectors

def adj_to_edgelist(adj_df, output_dir = None):
# convert df from adjacency to edgelist for csgraph import
    adj_df_copy = adj_df.copy()
    adj_df_copy.values[tuple([np.arange(len(adj_df_copy))]*2)] = np.nan
    edge_df = adj_df_copy.stack().reset_index()
    if output_dir != None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        edge_df.to_csv(f'{output_dir}/edgelist.txt', sep = '\t', index = 0, header = None)
        print(f'edgelist.txt has been saved.')
    return edge_df

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

def network_embedding_fast(edgelist_data, max_epoch = 100, learning_rate = 0.1, negative_ratio = 0.15, tol_samples = 75, output_dir = None, name_spec = ''):
    '''
    edgelist_path: path to the edgelist for embedding
    name_spec: any additional info as str to add for saving the embedding df
    '''
    G = cg.read_edgelist(edgelist_data, directed = False, sep = '\t')
    ggvec_model = nodevectors.GGVec(n_components = 64, max_epoch = max_epoch, learning_rate = learning_rate, 
                                    negative_ratio = negative_ratio, tol_samples = tol_samples) 
    embeddings = ggvec_model.fit_transform(G)
    emb_df = pd.DataFrame(embeddings, index = ggvec_model.model.keys())
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        emb_df.to_csv(f'{output_dir}/embedded_ggvec_{name_spec}.csv')
        print('embedding data saved')
    return emb_df

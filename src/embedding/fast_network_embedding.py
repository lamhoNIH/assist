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

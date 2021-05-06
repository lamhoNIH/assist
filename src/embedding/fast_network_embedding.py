import pandas as pd
import numpy as np
import os
from sys import platform 
import csrgraph as cg
import nodevectors

# def adj_to_edgelist(adj_df, output_dir = None):
# # convert df from adjacency to edgelist for csgraph import
#     adj_df_copy = adj_df.copy()
#     del adj_df
#     adj_df_copy = adj_df_copy.astype('float16')
#     adj_df_copy.values[tuple([np.arange(len(adj_df_copy))]*2)] = np.nan
#     edge_df = adj_df_copy.stack().reset_index()
#     del adj_df_copy
#     if output_dir != None:
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         edge_df.to_csv(f'{output_dir}/edgelist.txt', sep = '\t', index = 0, header = None)
#         print(f'edgelist.txt has been saved.')

def network_embedding_fast(tom_df, max_epoch = 100, learning_rate = 0.1, negative_ratio = 0.15, tol_samples = 75, output_path = None):
    '''
    tom_df: tom df for embedding
    name_spec: any additional info as str to add for saving the embedding df
    '''
    G = cg.read_adjacency(tom_df, directed = False) 
    print('read_adjacency success')
    ggvec_model = nodevectors.GGVec(n_components = 64, max_epoch = max_epoch, learning_rate = learning_rate, 
                                    negative_ratio = negative_ratio, tol_samples = tol_samples) 
    embeddings = ggvec_model.fit_transform(G.mat)
    print('fit_transform success')
    emb_df = pd.DataFrame(embeddings, index = G.names)
    del G # free up some space
    if output_path:
        emb_df.to_csv(output_path)
        print('embedding data saved')
    # if output_path:
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_dir)
    #     emb_df.to_csv(f'{output_path}/embedded_ggvec_epoch={max_epoch}_alpha={learning_rate}_sample={tol_samples}.csv')
    #     print('embedding data saved')
    return emb_df
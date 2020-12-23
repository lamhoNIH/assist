import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
from sys import platform
from ..preproc.community_data import CommunityData
from ..preproc.deseq_data import DESeqData
from functools import reduce
import numpy as np

prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'

def subset_network(network_df, weight_min, weight_max, num_edges = None):
    '''
    A function to subset a network using weight cutoff
    if num_edges is None, the returned subset network will have the edges outside the weight limits = 0 and all the nodes are preserved
    if num_edges is a number, the returned subset network will have the number of edges same as num_edges and nodes with degrees = 0 are removed
    '''
    subset = network_df[(network_df > weight_min) & (network_df < weight_max)]
    subset_edge = subset.stack() # convert from wide to long to determine # edges left
    # return the subnetwork with all the edges and nodes preserved
    if num_edges == None:
        print('Number of edges left:',len(subset_edge)/2)
        subset = subset.fillna(0)
        G = nx.convert_matrix.from_pandas_adjacency(subset)
        return subset, G
    # return the subnetwork with edges < cutoff and nodes with degree = 0 removed
    else:
        sorted_subset_edge = subset_edge.sort_values(0, ascending = False).reset_index()
        if num_edges > len(sorted_subset_edge)*2:
            print(f'not enough edges to filter with at the cutoff of {num_edges}')
        else:
            sorted_subset_edge_filtered = sorted_subset_edge.iloc[:num_edges*2, :]
            new_subset_adj = sorted_subset_edge_filtered.pivot(index = 'level_0', columns = 'level_1').fillna(0) # convert from edgelist back to adjacency matrix
            new_subset_adj.columns = new_subset_adj.columns.droplevel()
            G = nx.convert_matrix.from_pandas_adjacency(new_subset_adj)
        return new_subset_adj, G
    
def get_module_df(network_df, community_df, cluster):
    cluster_genes = community_df[community_df.louvain_label == cluster].id
    cluster_tom = network_df[cluster_genes]
    cluster_tom = cluster_tom[cluster_tom.index.isin(cluster_genes)]
    return cluster_tom

def plot_module_hist(adjacency_df, title, output_dir = None, comm_df = CommunityData.get_comm_df()):
    plt.hist(comm_df[comm_df.id.isin(adjacency_df.columns)].louvain_label) # show the distributions of the nodes after subsetting
    plt.title(title)
    plt.xlabel('module id')
    plt.ylabel('number of genes')
    x_min = comm_df.louvain_label.min()
    x_max = comm_df.louvain_label.max()
    plt.xticks(np.arange(x_min, x_max + 1, 1.0))
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f'{output_dir}/{title}.png', bbox_inches = 'tight')
        print(f'Figure {title} has been saved.')
    plt.close()


def get_subnetwork1(module, num_genes, min_weight, network_df, comm_df = CommunityData.get_comm_df(), deseq = DESeqData.get_deseq(), plot_hist = True, output_dir = None):
    '''This function subset the whole network by taking the top num_genes of DE genes(nodes) from module 4 and same number of genes(nodes) from 1 of the non-DE module in the original network
    module: the non-DE module to choose from
    num_genes: number of genes to subset from the two modules
    network_df: whole network tom file
    comm_df: louvain community label file
    deseq: DE file
    return subnetwork with edges joined together as an adjacency df
    '''
    module4_tom = pd.read_csv(prefix + '/Shared drives/NIAAA_ASSIST/Data/other data/cluster4_TOM.csv', index_col = 0)
    other_module_tom = get_module_df(network_df, comm_df, module)
    m4_top100_nodes = deseq[deseq.id.isin(module4_tom.columns)][['id', 'abs_log2FC']].sort_values('abs_log2FC', ascending = False).reset_index(drop = True)[:num_genes]['id']
    random.seed(1)
    other_module_nodes = random.sample(other_module_tom.columns.tolist(), num_genes) # has randomness so I set a seed in the line above to remove the randomness
    G_sub_list = []
    edges = 0
    for gene in m4_top100_nodes: # iterate through the nodes
        gene_subnet = network_df[gene][network_df[gene] > min_weight] # set weight to choose neighbors from the whole network to could get nodes from other modules as well
        gene_edgelist = pd.DataFrame({'source':gene, 'target':gene_subnet.index, 'weight':gene_subnet.values})
        edges += len(gene_subnet)
        G_sub = nx.convert_matrix.from_pandas_edgelist(gene_edgelist, 'source', 'target', 'weight') # convert from edgelist to graph
        G_sub_list.append(G_sub)

    for gene in other_module_nodes:
        gene_subnet = network_df[gene][network_df[gene] > min_weight]
        gene_edgelist = pd.DataFrame({'source':gene, 'target':gene_subnet.index, 'weight':gene_subnet.values})
        edges += len(gene_subnet)
        G_sub = nx.convert_matrix.from_pandas_edgelist(gene_edgelist, 'source', 'target', 'weight')
        G_sub_list.append(G_sub)
    print('Number of edges:',edges)
    
    G_joined = reduce(lambda x,y:nx.compose(x, y), G_sub_list)
    joined_df = nx.convert_matrix.to_pandas_adjacency(G_joined)
    if (plot_hist == True) & (output_dir == None):
        print('Must have an output dir to save the histogram')
    if plot_hist == True:
        plot_module_hist(joined_df, f'num_genes={num_genes},min_weight={min_weight}', output_dir)
    return G_joined, joined_df

def get_subnetwork2(num_genes, min_weight, network_df, comm_df = CommunityData.get_comm_df(), deseq = DESeqData.get_deseq(), output_dir = None, plot_hist = True):
    '''This function subset the whole network by taking the top num_genes DE from module 4 
    network_df: whole network tom file
    comm_df: louvain community label file
    deseq: DE file
    return subnetwork with edges joined together as an adjacency df
    '''
    module4_tom = pd.read_csv(prefix + '/Shared drives/NIAAA_ASSIST/Data/other data/cluster4_TOM.csv', index_col = 0)
    m4_top_nodes = deseq[deseq.id.isin(module4_tom.columns)][['id', 'abs_log2FC']].sort_values('abs_log2FC', ascending = False).reset_index(drop = True)[:num_genes]['id']

    G_sub_list = []
    edges = 0
    for gene in m4_top_nodes: # iterate through the nodes
        gene_subnet = network_df[gene][network_df[gene] > min_weight] # set weight to choose neighbors
        gene_edgelist = pd.DataFrame({'source':gene, 'target':gene_subnet.index, 'weight':gene_subnet.values})
        edges += len(gene_subnet)
        G_sub = nx.convert_matrix.from_pandas_edgelist(gene_edgelist, 'source', 'target', 'weight') # convert from edgelist to graph
        G_sub_list.append(G_sub)

    print('Number of edges:',edges)
    G_joined = reduce(lambda x,y:nx.compose(x, y), G_sub_list)
    joined_df = nx.convert_matrix.to_pandas_adjacency(G_joined)
    if (plot_hist == True) & (output_dir == None):
        print('Must have an output dir to save the histogram')
    if plot_hist == True:
        plot_module_hist(joined_df, f'num_genes={num_genes},min_weight={min_weight}', output_dir)
    return G_joined, joined_df
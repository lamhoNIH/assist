import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
from sys import platform
from ..preproc.deseq_data import DESeqData
from functools import reduce
import numpy as np

prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'

def subset_network(network_df, weight_min, weight_max, num_edges = None, subnetwork_dir = None):
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
        if subnetwork_dir != None:
            subset.to_csv(subnetwork_dir)
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
            if subnetwork_dir != None:
                subset.to_csv(subnetwork_dir)
        return new_subset_adj, G
    
def get_module_df(network_df, clusters, comm_df):
    clusters_genes = comm_df[comm_df.louvain_label.isin(clusters)].id
    clusters_tom = network_df[clusters_genes]
    clusters_tom = clusters_tom[clusters_tom.index.isin(clusters_genes)]
    return clusters_tom

def plot_module_hist(adjacency_df, title, comm_df, output_dir = None):
    module_num = len(comm_df.louvain_label.unique())
    plt.hist(comm_df[comm_df.id.isin(adjacency_df.columns)].louvain_label)
#              , bins = range(module_num)) # show the distributions of the nodes after subsetting
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
    plt.show()
    plt.close()

def get_subnetwork_by_DE(network_df, comm_df, abs_log2FC, pvalue = 0.05, min_weight = 0.012, deseq = DESeqData.get_deseq(), 
                         plot_hist = True, hist_dir = None, subnetwork_file = None):
    '''
    A method get subnetwork based on DE status. This method will take the DE with highest absoluate log2FC and then pull nodes with strong connection with the DE nodes
    '''
    genes_to_keep = deseq[(deseq.abs_log2FC > abs_log2FC) & (deseq.pvalue < pvalue)]['id']
    print('# DE:', len(genes_to_keep))
    G_sub_list = []
    edges = 0
    for gene in genes_to_keep: # iterate through the nodes
        gene_subnet = network_df[gene][network_df[gene] > min_weight] # set weight to choose neighbors from the whole network to could get nodes from other modules as well
        gene_edgelist = pd.DataFrame({'source':gene, 'target':gene_subnet.index, 'weight':gene_subnet.values})
        edges += len(gene_subnet)
        G_sub = nx.convert_matrix.from_pandas_edgelist(gene_edgelist, 'source', 'target', 'weight') # convert from edgelist to graph
        G_sub_list.append(G_sub)
    print('Number of edges:',edges)
    G_joined = reduce(lambda x,y:nx.compose(x, y), G_sub_list)
    joined_df = nx.convert_matrix.to_pandas_adjacency(G_joined)
    if (plot_hist == True) & (hist_dir == None):
        plot_module_hist(joined_df, f'abs_log2FC_{abs_log2FC},pvalue_{pvalue},min_weight_{min_weight}', comm_df)
        print('The histogram was not saved')
    if (plot_hist == True) & (hist_dir != None):
        plot_module_hist(joined_df, f'abs_log2FC_{abs_log2FC},pvalue_{pvalue},min_weight_{min_weight}', comm_df, hist_dir)
    if subnetwork_file != None:
        joined_df.to_csv(subnetwork_file + f'subnetwork_{abs_log2FC}_{pvalue}_{min_weight}.csv')
    return G_joined, joined_df



def get_subnetwork(deg_modules, num_genes, min_weight, network_df, comm_df, deseq, non_deg_modules= [], plot_hist = True, hist_dir = None, subnetwork_dir = None):
    '''This function subset the whole network by taking the top num_genes of DE genes(nodes) from module 4 and same number of genes(nodes) from 1 of the non-DE module in the original network
    deg_modules: a list of DEG modules to use
    num_genes: number of genes to subset from the two modules
    min_weight: weight cutoff
    network_df: whole network tom file
    comm_df: louvain community label file
    non_deg_modules: a list of non-DEG modules to use. If it's an empty list, then the function will only take nodes from DEG modules
    deseq: DE file
    return subnetwork with edges joined together as an adjacency df
    '''
    deg_module_tom = get_module_df(network_df, deg_modules, comm_df)
    if 'abs_log2FC' not in deseq:
        deseq['abs_log2FC'] = abs(deseq['log2FoldChange'])
    deg_module_nodes = deseq[deseq.id.isin(deg_module_tom.columns)][['id', 'abs_log2FC']].sort_values('abs_log2FC', ascending = False).reset_index(drop = True)[:num_genes]['id']
    
    G_sub_list = []
    edges = 0

    for gene in deg_module_nodes:
        gene_subnet = network_df[gene][network_df[gene] > min_weight]
        gene_edgelist = pd.DataFrame({'source':gene, 'target':gene_subnet.index, 'weight':gene_subnet.values})
        edges += len(gene_subnet)
        G_sub = nx.convert_matrix.from_pandas_edgelist(gene_edgelist, 'source', 'target', 'weight')
        G_sub_list.append(G_sub)
        
    if non_deg_modules != []: # if this list isn't empty, then find nodes in non_deg_modules and subselect them
        random.seed(1)
        non_deg_module_tom = get_module_df(network_df, non_deg_modules, comm_df)
        non_deg_module_nodes = random.sample(non_deg_module_tom.columns.tolist(), num_genes) # has randomness so I set a seed in the line above to remove the randomness
        for gene in non_deg_module_nodes: # iterate through the nodes
            gene_subnet = network_df[gene][network_df[gene] > min_weight] # set weight to choose neighbors from the whole network to could get nodes from other modules as well
            gene_edgelist = pd.DataFrame({'source':gene, 'target':gene_subnet.index, 'weight':gene_subnet.values})
            edges += len(gene_subnet)
            G_sub = nx.convert_matrix.from_pandas_edgelist(gene_edgelist, 'source', 'target', 'weight') # convert from edgelist to graph
            G_sub_list.append(G_sub)
    
    print('Number of edges:',edges)
    
    G_joined = reduce(lambda x,y:nx.compose(x, y), G_sub_list)
    joined_df = nx.convert_matrix.to_pandas_adjacency(G_joined)

    if (plot_hist == True) & (hist_dir == None):
        plot_module_hist(joined_df, f'deg_mod={deg_modules},non_deg_mod={non_deg_modules},num_genes={num_genes},min_weight={min_weight}', comm_df)
        print('The histogram was not saved')
    if (plot_hist == True) & (hist_dir != None):
        plot_module_hist(joined_df, f'deg_mod={deg_modules},non_deg_mod={non_deg_modules},num_genes={num_genes},min_weight={min_weight}', comm_df, hist_dir)
    if subnetwork_dir != None:
        joined_df.to_csv(os.path.join(subnetwork_dir, f'deg_mod={deg_modules},non_deg_mod={non_deg_modules},num_genes={num_genes},min_weight={min_weight}.csv'))
        
    network_name = f'module{deg_modules}_n_{non_deg_modules}_df'
    return G_joined, joined_df, network_name

def add_missing_genes(whole_network, subnetwork_df):
    '''A function to add back the genes that got cut out of the df when network was subselected'''
    subnetwork_df_copy = subnetwork_df.copy()
    id_diff = list(set(whole_network.columns) - set(subnetwork_df.columns))
    subnetwork_df_copy[id_diff] = 0
    rows_to_append = pd.DataFrame(0, columns = subnetwork_df_copy.columns, index = id_diff)
    subnetwork_complete = pd.concat([subnetwork_df_copy, rows_to_append])
    return subnetwork_complete
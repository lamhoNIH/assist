# backup functions removed from eda_functions.py not in the current modules
import sys
sys.path.append('../../src')
import os
import netcomp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
from matplotlib import gridspec
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
from .process_phenotype import *
from preproc.result import Result
from sklearn.metrics.pairwise import euclidean_distances as ed


def scale_free_validate(network_df, network_name):
    network_degree = network_df.sum()
    log_network_degree = np.log(network_degree)
    sorted_network_freq = round(log_network_degree, 2).value_counts().reset_index()
    sorted_network_freq[0] = np.log(sorted_network_freq[0])
    plt.figure(figsize = (4,4))
    plt.rcParams.update({'font.size': 15})
    plt.scatter(sorted_network_freq.index, sorted_network_freq[0])
    plt.xlabel('log(k)')
    plt.ylabel('log(pk)')
    plt.title(f'Scale-free check for \n {network_name}')
    plot_name = f'scale_free_validate_{network_name.replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(Result.getPath(), plot_name))
    plt.show() # This function needs plt.show() and plt.close() because other methods loop through the figures as subplots so they don't overlap. Each figure here is a whole plot
    plt.close()

def get_graph_distance(wholenetwork_np, network_np):
    dc_distance = netcomp.deltacon0(wholenetwork_np, network_np)
    ged_distance = netcomp.edit_distance(wholenetwork_np, network_np)
    return dc_distance, ged_distance

def plot_graph_distances(dc_distance_list, ged_distance_list, names):
    width = (len(dc_distance_list)+1)*2
    plt.figure(figsize=(width, 5))
    plt.rcParams.update({'font.size': 18})
    plt.subplot(1, 2, 1)
    plt.bar(names, dc_distance_list)
    plt.title('Deltacon distance')
    plt.xticks(rotation = 45, ha = 'right')

    plt.subplot(1, 2, 2)
    plt.bar(names, ged_distance_list)
    plt.title('GEM distance')
    plt.xlabel('Number of edges')
    plt.xticks(rotation = 45, ha = 'right')
    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(Result.getPath(), "plot_graph_distance.png"))
    plt.show()
    plt.close()

def plot_graph_distance(networks, network_names):

    dc_distance_list = []
    ged_distance_list = []
    names = []
    network1 = networks[0]
    ## compare all the network starting from the second to the first network (reduce run time)
    i = 1
    for network in networks[1:]:
        dc_distance_list.append(netcomp.deltacon0(network1.values, network.values))
        ged_distance_list.append(netcomp.edit_distance(network1.values, network.values))
        names.append(f'{network_names[0]} vs {network_names[i]}')
        i += 1
    ## pairwise combination (long run time)
    # for sub_network1, sub_network2 in combinations(zip(networks, network_names), 2):
    #     dc_distance_list.append(netcomp.deltacon0(sub_network1[0].values, sub_network2[0].values))
    #     ged_distance_list.append(netcomp.edit_distance(sub_network1[0].values, sub_network2[0].values))
    #     names.append(f'{sub_network1[1]} vs {sub_network2[1]}')
    width = len(networks)*2
    plt.figure(figsize=(width, 5))
    plt.rcParams.update({'font.size': 18})
    plt.subplot(1, 2, 1)
    plt.bar(names, dc_distance_list)
    plt.title('Deltacon distance')
    plt.xticks(rotation = 45, ha = 'right')

    plt.subplot(1, 2, 2)
    plt.bar(names, ged_distance_list)
    plt.title('GEM distance')
    plt.xlabel('Number of edges')
    plt.xticks(rotation = 45, ha = 'right')
    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(Result.getPath(), "plot_graph_distance.png"))
    plt.show()
    plt.close()

def cluster_jaccard(cluster_df1, cluster_df2, cluster_column, comparison_names, 
                    cutout_nodes = False, top=None, y_max = 1):
    '''
    plot jaccard pairwise comparison on the communities in 2 networks or the kmeans clusters in 2 network embeddings
    title: main title for the two subplots
    cluster_column: the column name of the cluster labels
    comparison_names: names of the groups in comparison
    cutout_nodes: if True, the nodes cut out in the smaller network/embedding is not included. If False, the cutout nodes will be in its own cluster for comparison
    top: top n comparison to show in the boxplot since it could be misleadingly small if we include all jaccard scores
    y_max to adjust y label
    # we're only interested in the modules that have majority of the matching nodes between 2 networks
    '''
    c1_list = []
    c2_list = []
    j_list = []
    if cutout_nodes == False:
        cluster_df2 = add_cutout_node_cluster(cluster_df1, cluster_df2, cluster_column)
        
    for c1 in cluster_df1[cluster_column].unique():
        for c2 in cluster_df2[cluster_column].unique():
            sub1 = cluster_df1[cluster_df1[cluster_column] == c1].index
            sub2 = cluster_df2[cluster_df2[cluster_column] == c2].index
            c1_list.append(c1)
            c2_list.append(c2)
            j_list.append(jaccard_similarity(sub1, sub2))

    jac_df = pd.DataFrame({'cluster1': c1_list, 'cluster2': c2_list, 'jaccard': j_list})
    jac_df = jac_df.pivot(index='cluster1', columns='cluster2', values='jaccard')
    sns.set(font_scale=1.25)
    sns.set_style('white')

    w = len(cluster_df2[cluster_column].unique())/1.3
    h = len(cluster_df1[cluster_column].unique())/2
    fig = plt.figure(figsize=(w, h))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # set the subplot width ratio
    ax0 = plt.subplot(gs[0])
    # plot heatmap for pairwise jaccard comparison
    sns.heatmap(jac_df, cmap='Reds', xticklabels=True, yticklabels=True)
    plt.xlabel(comparison_names[1])
    plt.ylabel(comparison_names[0])
    plt.title('Jaccard pairwise')
    plt.xticks(rotation=0)
    ax1 = plt.subplot(gs[1])
    # boxplot of jaccard distribution
    all_jac_values = jac_df.values.flatten()
    if top != None:
        sorted_jac_values = sorted(all_jac_values, reverse=True)
        g = sns.boxplot(x=None, y=sorted_jac_values[:top])
        g.set(ylim=(0, y_max))

    else:
        sns.boxplot(x=None, y=all_jac_values)
    plt.ylim(0, y_max)
    plt.title('Jaccard distribution')
#     plt.suptitle(f'{comparison_names[0]} vs {comparison_names[1]}')
    plt.subplots_adjust(top = 0.8, wspace = 1)
    plt.savefig(os.path.join(Result.getPath(), f'cluster_jaccard_{comparison_names[0]} vs {comparison_names[1]}_{cutout_nodes}.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def cluster_nmi(cluster_df1, cluster_df2, cluster_column):
    '''NMI to compare communities from the whole netowrk and the subnetwork or clusters from different network embeddings'''
    assert len(cluster_df1) >= len(cluster_df2), 'cluster_df1 must be greater than cluster_df2'
    num_cut_nodes = len(set(cluster_df1.id) - set(cluster_df2.id)) # number of nodes cut out in the subset
    sub1_plus_sub2 = pd.merge(cluster_df1, cluster_df2, left_on = 'id', right_on = 'id', how = 'left')
    num_cluster = cluster_df2[cluster_column].max() # determine how many clusters are present in the smaller network
    sub1_plus_sub2[f'{cluster_column}_y'] = sub1_plus_sub2[f'{cluster_column}_y'].fillna(num_cluster+1) # for the nodes that were cut out, give them a new community number
    return nmi(sub1_plus_sub2[f'{cluster_column}_x'], sub1_plus_sub2[f'{cluster_column}_y'])

def plot_cluster_nmi_comparison(cluster1_name, cluster1, cluster_list, cluster_column, comparison_names):
    plt.figure(figsize = (5,4))
    plt.rcParams.update({'font.size': 18})
    nmi_scores = []
    for cluster in cluster_list:
        nmi_scores.append(cluster_nmi(cluster1, cluster, cluster_column))
    plt.bar(comparison_names, nmi_scores)
    plt.ylabel('NMI')
    cluster_type = ['community' if cluster_column == 'louvain_label' else 'cluster']
    plt.title(f'NMI for {cluster_type[0]} comparison')
    plt.xticks(rotation = 45, ha = 'right')
    plt.savefig(os.path.join(Result.getPath(), f'plot_cluster_nmi_comparison_{cluster1_name}.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()
    
def cluster_nmi_v2(cluster_df1, cluster_df2, cluster_column):
    '''NMI to compare communities from the whole netowrk and the subnetwork or clusters from different network embeddings'''
    sub1_plus_sub2 = pd.merge(cluster_df1, cluster_df2, left_on = 'id', right_on = 'id', how = 'outer')
    max_cluster_num = max(sub1_plus_sub2[[f'{cluster_column}_x', f'{cluster_column}_y']].max())
    sub1_plus_sub2.fillna(max_cluster_num+1, inplace = True) # for the nodes that were cut out, give them a new community number
    return nmi(sub1_plus_sub2[f'{cluster_column}_x'], sub1_plus_sub2[f'{cluster_column}_y'])

def plot_cluster_nmi_comparison_v2(cluster1_list, cluster2_list, cluster1_names, cluster2_names, cluster_column):
    width = len(cluster1_list)*2
    plt.figure(figsize = (width,4))
    nmi_scores = []
    comparison_names = []
    for i in range(len(cluster1_list)):
        nmi_scores.append(cluster_nmi_v2(cluster1_list[i], cluster2_list[i], cluster_column))
        comparison_names.append(f'{cluster1_names[i]} vs {cluster2_names[i]}')
    plt.bar(comparison_names, nmi_scores)
    plt.ylabel('NMI')
    cluster_type = ['community' if cluster_column == 'louvain_label' else 'cluster']
    plt.title(f'NMI for {cluster_type[0]} comparison')
    plt.xticks(rotation = 45, ha = 'right')
    plt.savefig(os.path.join(Result.getPath(), f'plot_{cluster_type}_nmi_comparison.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()    


def plot_cluster_nmi_comparison_v3(cluster1_name, cluster1, cluster1_column, cluster2_list, cluster2_column, comparison_names):
    '''plot cluster_nmi_v3() results'''
    plt.figure(figsize = (5,4))
    plt.rcParams.update({'font.size': 18})
    nmi_scores = []
    for cluster2 in cluster2_list:
        nmi_scores.append(cluster_nmi_v3(cluster1, cluster1_column, cluster2, cluster2_column))
    plt.bar(comparison_names, nmi_scores)
    plt.ylabel('NMI')
    plt.title(f'NMI for cluster comparison')
    plt.xticks(rotation = 45, ha = 'right')
    plt.savefig(os.path.join(Result.getPath(), f'plot_cluster_nmi_comparison.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()

def gene_set_phenotype_corr(gene_sets, network_names, expression_meta_df, file_name):
    '''
    Plot correlation heatmap between critical gene sets and alcohol phenotypes
    (similar to cluster_phenotype_corr, cluster is replaced with a set of critical genes)
    '''
    i = 1
    length = len(gene_sets)
    empty_list = sum(1 for gene_set in gene_sets if len(gene_set) == 0)
    if length == empty_list:
        print('There is no overlapping critical genes between the critical gene sets')
        print(f'A suggested action is to change get_critical_gene_sets() parameters ratio and max_dist_ratio')
        return None
    
    empty_set_index = []
    non_empty_set_index = []
    for j, gene_set in enumerate(gene_sets):
        if len(gene_set) == 0:
            empty_set_index.append(j)
            
        else:
            non_empty_set_index.append(j)
            geneset_expression = expression_meta_df[gene_set].apply(pd.to_numeric)
            pca = PCA(n_components=1)
            pca_geneset_expression = pca.fit_transform(geneset_expression)
            # originally I used pd.corr() to get pairwise correlation matrix but since I need a separate calculation for correlation p value
            # I just used pearsonr and collected the results in lists. Making a df here isn't necessary anymore. 
            eigen_n_features = pd.DataFrame({'eigen': pca_geneset_expression.reshape(len(pca_geneset_expression), ),
    #                                          'BMI': expression_meta_df['BMI'], 
    #                                          'RIN': expression_meta_df['RIN'],
    #                                          'Age': expression_meta_df['Age'], 'PM!': expression_meta_df['PM!'],
    #                                          'Brain_pH': expression_meta_df['Brain_pH'],
    #                                          'Pack_yrs_1_pktperday_1_yr': expression_meta_df['Pack_yrs_1_pktperday_1_yr'],
                                             'AUDIT': expression_meta_df['AUDIT'],
                                             'Alcohol_intake_gmsperday': expression_meta_df['Alcohol_intake_gmsperday'],
                                             'Total_drinking_yrs': expression_meta_df['Total_drinking_yrs']})

            corr_list = []
#             p_list = []
#             corrected_p_list = []
            labels = []
            for col in eigen_n_features.columns[1:]:
                sub = eigen_n_features[['eigen', col]]
                sub = sub.dropna()
                corr_list.append(pearsonr(sub['eigen'], sub[col])[0])
#                 p_list.append(pearsonr(sub['eigen'], sub[col])[1])
#             corrected_p_list = multipletests(p_list, method ='fdr_bh')[1] # correct for multiple tests
            if i == 1:
                clusters_corr = pd.DataFrame({i: corr_list})
#                 clusters_pvalue = pd.DataFrame({i: corrected_p_list})
                i += 1

            else:
                clusters_corr[i] = corr_list
#                 clusters_pvalue[i] = corrected_p_list
                i += 1
    clusters_corr = clusters_corr.T.sort_index(ascending = False)
    clusters_corr = np.round(clusters_corr, 2)
#     clusters_pvalue = clusters_pvalue.T.sort_index(ascending = False)
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 18})
#     gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # set the subplot width ratio
#     # first subplot to show the correlation heatmap
#     ax0 = plt.subplot(gs[0])
#     sns.heatmap(clusters_corr, cmap='RdBu_r', annot = True,
#                 annot_kws = {'fontsize':12}, vmin=-1, vmax=1, xticklabels = eigen_n_features.columns[1:]) 
#     plt.xticks(rotation = 45, ha = 'right')
#     yticklabels = [network_names[index] for index in non_empty_set_index]
#     plt.yticks(np.arange(len(yticklabels))+0.5, labels=yticklabels, 
#                rotation = 0)
#     plt.ylabel('gene set id')
#     plt.title('Trait-critical gene set correlation')
#     # second subplot to show count of significant traits in each cluster. "Significant" here means adj p value < 0.2
#     ax1 = plt.subplot(gs[1])
#     sig_count = (clusters_pvalue < 0.2).sum(axis = 1) # count num of traits with p-adj < 0.2 in each cluster
#     plt.barh(sig_count.index, sig_count.values) # horizontal bar plot
#     plt.xlim(0,9) # there are 9 traits here so set the scale to between 0 and 9. change it if the # traits change
#     plt.ylabel('gene set id')
#     plt.xlabel('Trait count')
#     plt.yticks(np.arange(len(yticklabels)) +1, labels=yticklabels, 
#                rotation = 0)
#     plt.title('# significant traits')
#     plt.subplots_adjust(top = 1, bottom = 0.1)

    sns.heatmap(clusters_corr, cmap='RdBu_r', annot = True,
                annot_kws = {'fontsize':12}, vmin=-1, vmax=1, xticklabels = eigen_n_features.columns[1:]) 
    plt.xticks(rotation = 45, ha = 'right')
    yticklabels = [network_names[index] for index in non_empty_set_index]
    plt.yticks(np.arange(len(yticklabels))+0.5, labels=yticklabels, 
               rotation = 0)
    plt.ylabel('gene set id')
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    for index in empty_set_index:
        print(network_names[index], 'does not have critical genes in common between all 3 models')
    plt.savefig(os.path.join(Result.getPath(), f'gene_set_phenotype_corr_{file_name}.png'))
    plt.show()
    plt.close()
    
def get_closest_genes_jaccard(network, emb, gene_list, top_n, title):
    '''A function to compare how much closest genes are in common between a network and its embedding
    network: tom df
    emb: embedding df
    gene_list: a list of genes to query
    top_n: top n closest genes to the genes in gene_list
    title: title for the figure
    '''
    ed_data = ed(emb, emb)
    ed_df = pd.DataFrame(ed_data, index = emb.index, columns = emb.index)
    closest_genes1 = [] # find closest genes in the subnetwork
    closest_genes2 = [] # find closest genes in the embedding
    for gene in gene_list:
        closest_genes1.append(network[gene].sort_values(ascending = False)[:top_n].index)
        top_n_genes = ed_df[gene].sort_values()[1:top_n+1].index
        closest_genes2.append(top_n_genes)
    jac_list = []
    for i in range(len(closest_genes1)):
        jac_list.append(jaccard_similarity(closest_genes1[i], closest_genes2[i]))
#     xticks = le.inverse_transform(subnet1_edge.source.unique())
    plt.rcParams.update({'font.size':18})
    plt.bar(gene_list, jac_list)
    plt.ylim(0, 1)
    plt.ylabel('Jaccard similarity')
    plt.xlabel('gene')
    plt.xticks(rotation = 45, ha = 'right')
    plt.title(title)
    plt.show()
    plt.close()
    
    
def plot_dist(summary_df, sample_name, trait, summary_type):
    '''Plot distribution of some kind of summary table'''
    plt.rcParams.update({'font.size':18})
    sns.kdeplot(summary_df, label = sample_name)
    plt.title(trait)
    if summary_type == 'correlation':
        xlabel = 'Correlation coefficient'
    elif summary_type == 'significance':
        xlabel = '% significant genes'
    else:
        print('Summary type not recognized')
        xlabel = ''
    plt.xlabel(xlabel)
    plt.ylabel('Events')
    plt.legend()


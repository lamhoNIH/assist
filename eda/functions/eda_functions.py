import netcomp
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import gridspec
from scipy.stats import f_oneway
from sknetwork.clustering import Louvain



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
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(names, dc_distance_list)
    plt.title('Deltacon distance')
    plt.xlabel('Number of edges')
    plt.subplot(1, 2, 2)
    plt.bar(names, ged_distance_list)
    plt.title('GEM distance')
    plt.xlabel('Number of edges')
    plt.subplots_adjust(wspace=0.5)

def run_kmeans(embedding_df, n_clusters):
    '''Run k means on embedding df'''
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(embedding_df)
    k_mean_df = pd.DataFrame({'id':embedding_df.index, 'kmean_label':kmeans})
    return k_mean_df

def run_louvain(adjacency_df):
    # louvain communities
    louvain = Louvain(modularity = 'Newman')
    labels = louvain.fit_transform(adjacency_df.values) # using networkx community requires converting the df to G first and the original network takes very long but this method can work on df 
    louvain_df = pd.DataFrame({'id':adjacency_df.index, 'louvain_label':labels})
    return louvain_df

def jaccard_similarity(list1, list2):
    intersection = len(set(list1).intersection(list2))
    union = len(set(list1)) + len(set(list2)) - intersection
    return intersection / union

def add_cutout_node_cluster(cluster_df1, cluster_df2, cluster_column):
    cluster_df2[cluster_column] = cluster_df2[cluster_column].apply(lambda x:(x+1)) # add 1 to each of the cluster id 
    diff_nodes = set(cluster_df1.id.unique()) - set(cluster_df2.id.unique())
    cutout_node_cluster = pd.DataFrame({'id':list(diff_nodes), cluster_column:0}) # assign the nodes that were cut out in cluster 0 
    cluster_df2 = pd.concat([cutout_node_cluster, cluster_df2])
    return cluster_df2

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
    sns.set(font_scale=1.2)
    sns.set_style('white')

    w = len(cluster_df2[cluster_column].unique())/1.5
    h = len(cluster_df1[cluster_column].unique())/2
    fig = plt.figure(figsize=(w, h))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # set the subplot width ratio
    ax0 = plt.subplot(gs[0])
    # plot heatmap for pairwise jaccard comparison
    sns.heatmap(jac_df, cmap='Reds', xticklabels=True, yticklabels=True)
    plt.xlabel(comparison_names[1])
    plt.ylabel(comparison_names[0])
    plt.title('Jaccard pairwise cluster comparison')
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
    plt.suptitle(f'{comparison_names[0]} vs {comparison_names[1]}')


def get_module_sig_gene_perc(expression_meta_df_df, cluster_df, cluster_column, cluster, trait):
    '''
    A function to get the percentage of genes in a module that are significantly variable by trait
    '''
    module_genes = cluster_df[cluster_df[cluster_column] == cluster]['id'].tolist()
    module_expression = expression_meta_df_df[module_genes].apply(pd.to_numeric)

    module_expression = module_expression.assign(trait=expression_meta_df_df[f'{trait}'])

    # collect genes from the module with p < 0.05 based on 1-way ANOVA
    anova_sig_genes = []
    trait_category = module_expression['trait'].unique()
    for gene in module_genes:
        if f_oneway(*(module_expression[module_expression['trait'] == category][gene] for category in trait_category))[1] < 0.05:  # if p-value < 0.05, add the gene to list
            anova_sig_genes.append(gene)
    return round(100 * len(anova_sig_genes) / len(module_genes), 2)  # return the % of genes found significant by ANOVA

def plot_sig_perc(expression_meta_df_df, cluster_df, cluster_column, trait, network_name):
    sig_gene_perc = []
    clusters = cluster_df[cluster_column].unique()
    for cluster in clusters:
        sig_gene_perc.append(get_module_sig_gene_perc(expression_meta_df_df, cluster_df, cluster_column, cluster, trait))

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # set the subplot width ratio
    ax0 = plt.subplot(gs[0])
    plt.scatter(clusters, sig_gene_perc)
    plt.xlabel('cluster label')
    plt.ylabel('% significant genes')
    plt.title('% significant genes by cluster')
    ax1 = plt.subplot(gs[1])
    sns.boxplot(x = None, y = sig_gene_perc)
    plt.ylabel('% significant genes')
    plt.suptitle(f'% significant genes in {trait} by cluster from {network_name}')


def cluster_phenotype_corr(expression_meta_df, cluster_df, cluster_column, network_name):
    '''
    Plot correlation heatmap between modules/clusters and alcohol phenotypes
    '''
    clusters = cluster_df[cluster_column].unique()
    i = 1
    for cluster in clusters:
        cluster_genes = cluster_df[cluster_df[cluster_column] == cluster]['id'].tolist()
        cluster_expression = expression_meta_df[cluster_genes].apply(pd.to_numeric)
        pca = PCA(n_components=1)
        pca_cluster_expression = pca.fit_transform(cluster_expression)
        eigen_n_features = pd.DataFrame({'eigen': pca_cluster_expression.reshape(len(pca_cluster_expression), ),
                                         'BMI': expression_meta_df['BMI'], 'RIN': expression_meta_df['RIN'],
                                         'Age': expression_meta_df['Age'], 'PM!': expression_meta_df['PM!'],
                                         'Brain_pH': expression_meta_df['Brain_pH'],
                                         'Pack_yrs_1_pktperday_1_yr': expression_meta_df['Pack_yrs_1_pktperday_1_yr)'],
                                         'AUDIT': expression_meta_df['AUDIT'],
                                         'alcohol_intake_gmsperday': expression_meta_df['alcohol_intake_gmsperday'],
                                         'Total_drinking_yrs': expression_meta_df['Total_drinking_yrs'],
                                         'SR': expression_meta_df['SR']})

        if i == 1:
            clusters_corr = pd.DataFrame({cluster: eigen_n_features.corr()['eigen'][1:]})
            i += 1
        else:
            clusters_corr[cluster] = eigen_n_features.corr()['eigen'][1:]
    clusters_corr = clusters_corr.T

    sns.heatmap(clusters_corr.sort_index(), cmap='RdBu',
                vmin=-1, vmax=1)
    plt.ylabel('cluster id')
    plt.title(f'Trait cluster correlation for {network_name}')
    
def cluster_nmi(cluster_df1, cluster_df2, cluster_column):
    '''NMI to compare communities from the whole netowrk and the subnetwork'''
    assert len(cluster_df1) >= len(cluster_df2), 'cluster_df1 must be greater than cluster_df2'
    num_cut_nodes = len(set(cluster_df1.id) - set(cluster_df2.id)) # number of nodes cut out in the subset
    sub1_plus_sub2 = pd.merge(cluster_df1, cluster_df2, left_on = 'id', right_on = 'id', how = 'left')
    num_cluster = cluster_df2[cluster_column].max() # determine how many clusters are present in the smaller network
    sub1_plus_sub2[f'{cluster_column}_y'] = sub1_plus_sub2[f'{cluster_column}_y'].fillna(num_cluster+1) # for the nodes that were cut out, give them a new community number
    return nmi(sub1_plus_sub2[f'{cluster_column}_x'], sub1_plus_sub2[f'{cluster_column}_y'])

def plot_cluster_nmi_comparison(cluster1, cluster_list, cluster_column, comparison_names):
    plt.figure(figsize = (5,4))
    nmi_scores = []
    for cluster in cluster_list:
        nmi_scores.append(cluster_nmi(cluster1, cluster, cluster_column))
    plt.bar(comparison_names, nmi_scores)
    plt.xlabel('Edges')
    plt.ylabel('NMI')
    plt.title('NMI for cluster comparison')
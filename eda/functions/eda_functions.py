import netcomp
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
from matplotlib import gridspec
from scipy.stats import f_oneway
from sknetwork.clustering import Louvain
from statsmodels.stats.multitest import multipletests
from functions.process_phenotype import *
from scipy.stats import pearsonr
from sys import platform
import math

prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'
expression_meta = pd.read_csv(prefix + '/Shared drives/NIAAA_ASSIST/Data/expression_meta.csv',
                              low_memory = False)

def scale_free_validate(network_df, network_name):
    network_degree = network_df.sum()
    log_network_degree = np.log(network_degree)
    sorted_network_freq = round(log_network_degree,1).value_counts().reset_index()
    sorted_network_freq[0] = np.log(sorted_network_freq[0])
    plt.scatter(sorted_network_freq.index, sorted_network_freq[0])
    plt.xlabel('log(k)')
    plt.ylabel('log(pk)')
    plt.title(f'Scale-free check for {network_name}')
    plt.show()
    plt.close();
    
def plot_gene_cnt_each_cluster(cluster_dfs, cluster_column, network_names):
    '''
    bar graphs to show # genes in each cluster
    cluster_dfs: a list of cluster dfs with id and cluster assignment
    cluster_column: cluster type, louvain or k means
    network_names: names to show in the subplot titles
    '''
    h = math.ceil(len(cluster_dfs)/3)
    plt.figure(figsize = (16,h*4))
    for i, cluster_df in enumerate(cluster_dfs):       
        plt.subplot(h, 3, i+1)
        plt.bar(cluster_df[cluster_column].value_counts().index, cluster_df[cluster_column].value_counts().values)
        plt.ylabel('# genes')
        plt.xlabel('Cluster id')
        plt.title(f'# genes in each community for {network_names[i]}')
        plt.subplots_adjust(wspace = 0.3)
        
        
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

def run_louvain(adjacency_df, resolution = 1, n_aggregations = -1):
    # louvain communities
    louvain = Louvain(modularity = 'Newman', resolution = resolution, n_aggregations  = n_aggregations)
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


def get_module_sig_gene_perc(expression_meta_df, cluster_df, cluster_column, cluster, trait):
    '''
    A function to get the percentage of genes in a module that are significantly variable by trait
    '''
    module_genes = cluster_df[cluster_df[cluster_column] == cluster]['id'].tolist()
    module_expression = expression_meta_df[module_genes].apply(pd.to_numeric)

    module_expression = module_expression.assign(trait=expression_meta_df[f'{trait}'])

    # collect genes from the module with p < 0.05 based on 1-way ANOVA
    anova_sig_genes = []
    trait_category = module_expression['trait'].unique()
    for gene in module_genes:
        if f_oneway(*(module_expression[module_expression['trait'] == category][gene] for category in trait_category))[1] < 0.05:  # if p-value < 0.05, add the gene to list
            anova_sig_genes.append(gene)
    return round(100 * len(anova_sig_genes) / len(module_genes), 2)  # return the % of genes found significant by ANOVA

def plot_sig_perc(cluster_df, cluster_column, network_name):
    '''
    A function to iterate through the clusters to get % significant genes in each clusters for each trait and show the results in a heatmap and barplot
    '''
    audit_subset = get_expression_by_audit()
    liver_class_subset = get_liver_class()
    alc_perday_subset = get_expression_by_alcohol_perday()
    drinking_yr_subset = get_expression_by_drinking_yrs()
    smoke_freq_subset = get_smoke_freq()
    traits = ['audit_category', 'Liver_class', 'alcohol_intake_category', 'drinking_yrs_category', 'Smoking_frequency']
    for i, subset in enumerate([audit_subset, liver_class_subset, alc_perday_subset, 
                                drinking_yr_subset, smoke_freq_subset]):
        sig_gene_perc = []
        clusters = cluster_df[cluster_column].unique()
        for cluster in clusters:
            sig_gene_perc.append(get_module_sig_gene_perc(subset, cluster_df, cluster_column, cluster, traits[i]))
        if i == 0:
            cluster_sig_perc = pd.DataFrame({traits[i]: sig_gene_perc})
        else:
            cluster_sig_perc[traits[i]] = sig_gene_perc
        
    cluster_sig_perc = cluster_sig_perc.sort_index(ascending = False)
    fig = plt.figure(figsize=(17, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # set the subplot width ratio
    # first subplot to show the correlation heatmap
    ax0 = plt.subplot(gs[0])
    sns.heatmap(cluster_sig_perc, cmap='Reds',
                vmin=0, vmax=100) 
    plt.xticks(rotation = 45, ha = 'right')
    plt.ylabel('cluster id')
    plt.title('% significant genes by cluster')
    # second subplot to show count of significant traits in each cluster. "Significant" here means adj p value < 0.2
    ax1 = plt.subplot(gs[1])
    sig_count = cluster_sig_perc[cluster_sig_perc > 5].count(axis = 1).values # count num of traits with significant gene % > 5 in each cluster
    plt.barh(cluster_sig_perc.index, sig_count) # horizontal bar plot
    plt.yticks(cluster_sig_perc.index, cluster_sig_perc.index)
    plt.xticks(np.arange(0, max(sig_count)+1, 1))
    plt.ylabel('cluster id')
    plt.xlabel('# Trait with >5% significant genes')
    plt.title('Number of significant traits each cluster')
    plt.suptitle(f'% significant genes for each trait for {network_name}', fontsize = 16)
    
    

def cluster_phenotype_corr(cluster_df, cluster_column, network_name, expression_meta_df = expression_meta):
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
        # originally I used pd.corr() to get pairwise correlation matrix but since I need a separate calculation for correlation p value
        # I just used pearsonr and collected the results in lists. Making a df here isn't necessary anymore. 
        eigen_n_features = pd.DataFrame({'eigen': pca_cluster_expression.reshape(len(pca_cluster_expression), ),
                                         'BMI': expression_meta_df['BMI'], 
#                                          'RIN': expression_meta_df['RIN'],
                                         'Age': expression_meta_df['Age'], 'PM!': expression_meta_df['PM!'],
                                         'Brain_pH': expression_meta_df['Brain_pH'],
                                         'Pack_yrs_1_pktperday_1_yr': expression_meta_df['Pack_yrs_1_pktperday_1_yr)'],
                                         'AUDIT': expression_meta_df['AUDIT'],
                                         'alcohol_intake_gmsperday': expression_meta_df['alcohol_intake_gmsperday'],
                                         'Total_drinking_yrs': expression_meta_df['Total_drinking_yrs'],
                                         'SR': expression_meta_df['SR']})

        corr_list = []
        p_list = []
        corrected_p_list = []
        labels = []
        for col in eigen_n_features.columns[1:]:
            sub = eigen_n_features[['eigen', col]]
            sub = sub.dropna()
            corr_list.append(pearsonr(sub['eigen'], sub[col])[0])
            p_list.append(pearsonr(sub['eigen'], sub[col])[1])
        corrected_p_list = multipletests(p_list, method ='fdr_bh')[1] # correct for multiple tests
        if i == 1:
            clusters_corr = pd.DataFrame({cluster: corr_list})
            clusters_pvalue = pd.DataFrame({cluster: corrected_p_list})
            i += 1

        else:
            clusters_corr[cluster] = corr_list
            clusters_pvalue[cluster] = corrected_p_list

    clusters_corr = clusters_corr.T.sort_index(ascending = False)
    clusters_pvalue = np.round(clusters_pvalue, 2)
    clusters_pvalue = clusters_pvalue.T.sort_index(ascending = False)

    fig = plt.figure(figsize=(17, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # set the subplot width ratio
    # first subplot to show the correlation heatmap
    ax0 = plt.subplot(gs[0])
    sns.heatmap(clusters_corr, cmap='RdBu_r', annot = clusters_pvalue,
                annot_kws = {'fontsize':12}, vmin=-1, vmax=1, xticklabels = eigen_n_features.columns[1:]) 
    plt.xticks(rotation = 45, ha = 'right')
    plt.ylabel('cluster id')
    plt.title('Trait cluster correlation')
    # second subplot to show count of significant traits in each cluster. "Significant" here means adj p value < 0.2
    ax1 = plt.subplot(gs[1])
    sig_count = (clusters_pvalue < 0.2).sum(axis = 1).values # count num of traits with p-adj < 0.2 in each cluster
    plt.barh(clusters_pvalue.index, sig_count) # horizontal bar plot
    plt.yticks(clusters_pvalue.index, clusters_pvalue.index)
    plt.xticks(np.arange(0, max(sig_count)+1, 1))
    plt.ylabel('cluster id')
    plt.xlabel('Trait count')
    plt.title('Number of significant traits each cluster')
    plt.suptitle(f'Trait cluster correlation for {network_name}', fontsize = 16)
    
def cluster_nmi(cluster_df1, cluster_df2, cluster_column):
    '''NMI to compare communities from the whole netowrk and the subnetwork or clusters from different network embeddings'''
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
    cluster_type = ['community' if cluster_column == 'louvain_label' else 'cluster']
    plt.title(f'NMI for {cluster_type[0]} comparison')
import os
import netcomp
import pandas as pd
import math
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
from scipy.stats import pearsonr
from .process_phenotype import *
from ..preproc.result import Result
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
    
def plot_gene_cnt_each_cluster(cluster_dfs, cluster_column, network_names):
    '''
    bar graphs to show # genes in each cluster
    cluster_dfs: a list of cluster dfs with id and cluster assignment
    cluster_column: cluster type, louvain or k means
    network_names: names to show in the subplot titles
    '''
    plt.rcParams.update({'font.size': 18})
    h = math.ceil(len(cluster_dfs)/3)
    plt.figure(figsize = (16,h*4))
    for i, cluster_df in enumerate(cluster_dfs):       
        plt.subplot(h, 3, i+1)
        plt.bar(cluster_df[cluster_column].value_counts().index, cluster_df[cluster_column].value_counts().values)
        plt.ylabel('# genes')
        plt.xlabel('Cluster id')
        plt.title(network_names[i])
        plt.subplots_adjust(wspace = 0.3)
    plt.tight_layout()    
    plt.savefig(os.path.join(Result.getPath(), "plot_gene_cnt_each_cluster.png"))
    plt.show()
    plt.close()

def plot_gene_cnt_each_cluster_v2(cluster_df, cluster_column, network_name, name_spec = ''):
    '''
    bar graphs to show # genes in each cluster
    cluster_dfs: a list of cluster dfs with id and cluster assignment
    cluster_column: cluster type, louvain or k means
    network_names: names to show in the subplot titles
    '''
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize = (6,4))
    plt.bar(cluster_df[cluster_column].value_counts().index, cluster_df[cluster_column].value_counts().values)
    plt.ylabel('# genes')
    plt.xlabel('Cluster id')
    plt.title(network_name)
    plt.tight_layout()
    plt.savefig(os.path.join(Result.getPath(), f"plot_gene_cnt_each_cluster{name_spec}.png"))
    plt.show()
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

def run_louvain2(adjacency_np, ajacency_idx, resolution = 1, n_aggregations = -1):
    # louvain communities
    louvain = Louvain(modularity = 'Newman', resolution = resolution, n_aggregations  = n_aggregations)
    labels = louvain.fit_transform(adjacency_np) # using networkx community requires converting the df to G first and the original network takes very long but this method can work on df 
    del louvain
    louvain_df = pd.DataFrame({'id':ajacency_idx, 'louvain_label':labels})
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

def cluster_jaccard_v2(cluster_df1, cluster_df2, cluster1_column, cluster2_column, comparison_names, 
                       top=None, y_max = 1):
    '''
    plot jaccard pairwise comparison on the communities in 2 networks or the kmeans clusters in 2 network embeddings
    title: main title for the two subplots
    cluster1_column: the column name of the cluster labels in cluster_df1
    cluster2_column: the column name of the cluster labels in cluster_df2
    comparison_names: names of the groups in comparison
    top: top n comparison to show in the boxplot since it could be misleadingly small if we include all jaccard scores
    y_max to adjust y label
    # we're only interested in the modules that have majority of the matching nodes between 2 networks
    '''
    c1_list = []
    c2_list = []
    j_list = []

    for c1 in cluster_df1[cluster1_column].unique():
        for c2 in cluster_df2[cluster2_column].unique():
            sub1 = cluster_df1[cluster_df1[cluster1_column] == c1].index
            sub2 = cluster_df2[cluster_df2[cluster2_column] == c2].index
            c1_list.append(c1)
            c2_list.append(c2)
            j_list.append(jaccard_similarity(sub1, sub2))

    jac_df = pd.DataFrame({'cluster1': c1_list, 'cluster2': c2_list, 'jaccard': j_list})
    jac_df = jac_df.pivot(index='cluster1', columns='cluster2', values='jaccard')
    sns.set(font_scale=1.25)
    sns.set_style('white')

    fig = plt.figure(figsize=(8,4))
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
    plt.savefig(os.path.join(Result.getPath(), f'cluster_jaccard_{comparison_names[0]} vs {comparison_names[1]}.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()
    
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

def plot_sig_perc(cluster_df, cluster_column, network_name, expression_meta_df, output_sig_df = False):
    '''
    A function to iterate through the clusters to get % significant genes in each clusters for each trait and show the results in a heatmap and barplot
    '''
    audit_subset = get_expression_by_audit(expression_meta_df)
    liver_class_subset = get_liver_class(expression_meta_df)
    alc_perday_subset = get_expression_by_alcohol_perday(expression_meta_df)
    drinking_yr_subset = get_expression_by_drinking_yrs(expression_meta_df)
#     smoke_freq_subset = get_smoke_freq(expression_meta_df)
#     traits = ['audit_category', 'Liver_class', 'alcohol_intake_category', 'drinking_yrs_category', 'Smoking_frequency']
    traits = ['audit_category', 'Liver_class', 'alcohol_intake_category', 'drinking_yrs_category']
#     for i, subset in enumerate([audit_subset, liver_class_subset, alc_perday_subset, 
#                                 drinking_yr_subset, smoke_freq_subset]):
    for i, subset in enumerate([audit_subset, liver_class_subset, alc_perday_subset, drinking_yr_subset]):
        sig_gene_perc = []
        clusters = cluster_df[cluster_column].unique()
        for cluster in clusters:
            sig_gene_perc.append(get_module_sig_gene_perc(subset, cluster_df, cluster_column, cluster, traits[i]))
        if i == 0:
            cluster_sig_perc = pd.DataFrame({traits[i]: sig_gene_perc})
        else:
            cluster_sig_perc[traits[i]] = sig_gene_perc
    cluster_sig_perc.index = clusters    
    cluster_sig_perc = cluster_sig_perc.sort_index(ascending = False)
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 18})
#     gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])  # set the subplot width ratio
#     # first subplot to show the correlation heatmap
#     ax0 = plt.subplot(gs[0])
    sns.heatmap(cluster_sig_perc, cmap='Reds',
                vmin=0, vmax=100) 
    plt.xticks(rotation = 45, ha = 'right')
    plt.ylabel('cluster id')
    plt.title('% significant genes by cluster')
#     # second subplot to show count of significant traits in each cluster. "Significant" here means adj p value < 0.2
#     ax1 = plt.subplot(gs[1])
#     sig_count = cluster_sig_perc[cluster_sig_perc > 5].count(axis = 1).values # count num of traits with significant gene % > 5 in each cluster
#     plt.barh(cluster_sig_perc.index, sig_count) # horizontal bar plot
#     plt.xlim(0,5) # there are 5 traits here so set the scale to between 0 and 5. change it if the # traits change
#     plt.yticks(cluster_sig_perc.index, cluster_sig_perc.index)

#     plt.ylabel('cluster id')
#     plt.xlabel('# Trait with >5% significant genes')
#     plt.title('Number of significant traits each cluster')
#     plt.suptitle(f'% significant genes for each trait for {network_name}', fontsize = 22)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout()
    plt.savefig(os.path.join(Result.getPath(), f'plot_sig_perc_{network_name}.png'))
    plt.show()
    plt.close()
    if output_sig_df == True:
        return cluster_sig_perc

def cluster_phenotype_corr(cluster_df, cluster_column, network_name, expression_meta_df, output_corr_df = False):
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
#                                          'BMI': expression_meta_df['BMI'], 
#                                          'RIN': expression_meta_df['RIN'],
#                                          'Age': expression_meta_df['Age'], 'PM!': expression_meta_df['PM!'],
#                                          'Brain_pH': expression_meta_df['Brain_pH'],
#                                          'Pack_yrs_1_pktperday_1_yr': expression_meta_df['Pack_yrs_1_pktperday_1_yr'],
                                         'AUDIT': expression_meta_df['AUDIT'],
                                         'Alcohol_intake_gmsperday': expression_meta_df['Alcohol_intake_gmsperday'],
                                         'Total_drinking_yrs': expression_meta_df['Total_drinking_yrs']})

        corr_list = []
#         p_list = []
#         corrected_p_list = []
        labels = []
        for col in eigen_n_features.columns[1:]:
            sub = eigen_n_features[['eigen', col]]
            sub = sub.dropna()
            corr_list.append(pearsonr(sub['eigen'], sub[col])[0])
#             p_list.append(pearsonr(sub['eigen'], sub[col])[1])
#         corrected_p_list = multipletests(p_list, method ='fdr_bh')[1] # correct for multiple tests
        if i == 1:
            clusters_corr = pd.DataFrame({cluster: corr_list})
#             clusters_pvalue = pd.DataFrame({cluster: corrected_p_list})
            i += 1

        else:
            clusters_corr[cluster] = corr_list
#             clusters_pvalue[cluster] = corrected_p_list

    clusters_corr = clusters_corr.T.sort_index(ascending = False)
    clusters_corr = np.round(clusters_corr, 2)
#     clusters_pvalue = clusters_pvalue.T.sort_index(ascending = False)

    fig = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 18})
    clusters_corr.columns = eigen_n_features.columns[1:]
#     gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # set the subplot width ratio
#     # first subplot to show the correlation heatmap
#     ax0 = plt.subplot(gs[0])
    sns.heatmap(clusters_corr, cmap='RdBu_r', annot = True,
                annot_kws = {'fontsize':12}, vmin=-1, vmax=1, xticklabels = eigen_n_features.columns[1:]) 
    plt.xticks(rotation = 45, ha = 'right')
    plt.yticks(rotation = 0)
    plt.ylabel('cluster id')
#     plt.title('Trait cluster correlation')
    # second subplot to show count of significant traits in each cluster. "Significant" here means adj p value < 0.2
#     ax1 = plt.subplot(gs[1])
#     sig_count = (clusters_pvalue < 0.2).sum(axis = 1) # count num of traits with p-adj < 0.2 in each cluster
#     plt.barh(sig_count.index, sig_count.values) # horizontal bar plot
#     plt.xlim(0,3.5) # there are 9 traits here so set the scale to between 0 and 9. change it if the # traits change
#     plt.yticks(clusters_pvalue.index, clusters_pvalue.index)
#     plt.ylabel('cluster id')
#     plt.xlabel('Trait count')
#     plt.title('Number of significant traits each cluster')
#     plt.suptitle(f'Trait cluster correlation for {network_name}', fontsize = 22)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(Result.getPath(), f'cluster_phenotype_corr_{network_name}.png'))
    plt.show()
    plt.close()
    if output_corr_df == True:
        return clusters_corr
    
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

def cluster_nmi_v3(cluster_df1, cluster1_column, cluster_df2, cluster2_column):
    '''NMI to compare communities from the whole netowrk and the subnetwork or clusters from different network embeddings'''
    sub1_plus_sub2 = pd.merge(cluster_df1, cluster_df2, left_on = 'id', right_on = 'id', how = 'outer')
    max_cluster_num = max(sub1_plus_sub2[[cluster1_column, cluster2_column]].max())
    sub1_plus_sub2[cluster2_column].fillna(max_cluster_num+1, inplace = True) # for the nodes that were cut out, give them a new community number
    return nmi(sub1_plus_sub2[cluster1_column], sub1_plus_sub2[cluster2_column])

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
    
def cluster_DE_perc(cluster_df, cluster_column, network_name, deseq):
    '''
    A function to plot 2 heatmaps to show % of differential genes in each cluster
    '''
    if 'abs_log2FC' not in deseq.columns:
        deseq['abs_log2FC'] = abs(deseq['log2FoldChange'])
    cutoff_index = int(len(deseq)*0.02)
    cutoff = deseq['abs_log2FC'].sort_values(ascending = False).reset_index(drop = True)[cutoff_index]
    num_up_impact = (deseq.log2FoldChange > cutoff).sum()
    num_down_impact = (deseq.log2FoldChange < -cutoff).sum()
    clusters = []
    up_impact_perc = []
    down_impact_perc = []
    for cluster in cluster_df[cluster_column].unique():
        cluster_genes = cluster_df[cluster_df[cluster_column] == cluster].id
        num_up_in_module = (deseq[deseq.id.isin(cluster_genes)]['log2FoldChange'] > cutoff).sum()
        num_down_in_module = (deseq[deseq.id.isin(cluster_genes)]['log2FoldChange'] < -cutoff).sum()

        clusters.append(cluster)
        up_impact_perc.append(100*num_up_in_module/num_up_impact)
        down_impact_perc.append(100*num_down_in_module/num_down_impact)
    cluster_DE_perc = pd.DataFrame({'cluster':clusters, '% up': up_impact_perc, '% down': down_impact_perc}) 
    cluster_DE_perc = cluster_DE_perc.sort_values('cluster', ascending = False)
    sns.set(font_scale=1.2)
    sns.set_style('white')
    if len(cluster_DE_perc) < 3:
        h = len(cluster_DE_perc)/1.5
    else:
        h = len(cluster_DE_perc)/3
    plt.figure(figsize = (4,h))
    plt.subplot(1,2,1)
    sns.heatmap(np.array([cluster_DE_perc['% up']]).T, xticklabels = ['% up'], yticklabels = cluster_DE_perc['cluster'], 
                cmap = 'Reds', vmin = 0, vmax = 100)    
    plt.ylabel('cluster id')
    plt.yticks(rotation=0)    
    plt.subplot(1,2,2)
    sns.heatmap(np.array([cluster_DE_perc['% down']]).T, xticklabels = ['% down'], yticklabels = cluster_DE_perc['cluster'], 
                cmap = 'Blues', vmin = 0, vmax = 100) 
    plt.yticks(rotation=0)
    # no one-size fits all so adjust the title location by # of clusters
    if len(cluster_DE_perc) < 7:
        top = 0.5 + (len(cluster_DE_perc) - 2)/13

    else:
        top = 0.85
    plt.subplots_adjust(wspace = 0.8, top = top)
    plt.suptitle(f'% DE in each cluster for {network_name}', fontsize = 22)
    plt.savefig(os.path.join(Result.getPath(), f'cluster_DE_perc_{network_name}.png'), bbox_inches = 'tight')
    plt.show()
    plt.close()

def permute_cluster_label(expression_meta_df, cluster_df1, cluster_df2, cluster1, cluster2, cluster_column, shuffle = 100):
    '''
    Given 2 cluster dfs, generate simulated random eigen gene expression for a cluster and get p values & correlation scores
    cluster1 and cluster2 need to match like in 2 community detections, only cluster 1 in network1 is compared with cluster 1 in network2 bc they're most correlated between the 2 networks
    '''
    p_w_random = []
    corr_w_random = []
    eigen1 = get_eigen_expression(cluster_df1, cluster1, cluster_column, expression_meta_df)
    for i in range(shuffle):
        eigen_random = get_random_expression(cluster_df2, cluster2, cluster_column, expression_meta_df)
        random_corr = pearsonr(eigen1.reshape(len(eigen1),), eigen_random.reshape(len(eigen_random),))
        p_w_random.append(random_corr[1])
        corr_w_random.append(random_corr[0])
    return np.array(p_w_random), np.array(corr_w_random)

def get_eigen_expression(cluster_df, cluster, cluster_column, expression_meta_df):
    '''Get eigen expression for a specific cluster in a cluster df'''
    cluster_genes = cluster_df[cluster_df[cluster_column] == cluster]['id'].tolist()
    cluster_expression = expression_meta_df[cluster_genes].apply(pd.to_numeric)
    pca = PCA(n_components=1)
    pca_cluster_expression = pca.fit_transform(cluster_expression)
    return pca_cluster_expression

def get_random_expression(cluster_df, cluster, cluster_column, expression_meta_df):
    '''Get eigen expression if the cluster membership is randomly assigned'''
    num_nodes = len(cluster_df[cluster_df[cluster_column] == cluster]) 
    random_genes = cluster_df.id.sample(num_nodes)
    random_expression = expression_meta_df[random_genes].apply(pd.to_numeric)
    pca = PCA(n_components=1)
    pca_random_expression = pca.fit_transform(random_expression)
    return pca_random_expression

def network_cluster_stability(cluster_df1, cluster_df2, cluster_column, expression_meta_df):
    '''
    Determine network cluster stability
    Network1 clusters (cluster_df1) is compared to network2 clusters (cluster_df2)
    cluster_column: louvain_label or kmean_label
    
    '''
    cluster_pairs = {}
    corr_cluster_df2 = []
    p_cluster_df2 = []
    # pairwise comparison to determine cluster correlation in network1 and network2
    # the clusters with the best correlation are then paired up as the ones to compare with random assignment
    for cluster2 in cluster_df2[cluster_column].unique(): # loop through all the clusters in cluster_df2
        p_list = []
        corr_list = []
        for cluster1 in cluster_df1[cluster_column].unique(): # loop through all the clusters in cluster_df1
            eigen1 = get_eigen_expression(cluster_df1, cluster1, cluster_column, expression_meta_df) # eigengene expression for cluster 1 in cluster_df1
            eigen2 = get_eigen_expression(cluster_df2, cluster2, cluster_column, expression_meta_df) # eigengene expression for cluster 2 in cluster_df2
            corr = pearsonr(eigen1.reshape(len(eigen1),), eigen2.reshape(len(eigen2),)) # get pearson correlation coef
            p_list.append(corr[1]) # add the p value only from the correlation coef
            corr_list.append(corr[0]) # add the correlation score
        min_p = min(p_list) # since we don't know which cluster in cluster_df1 should be compared to each cluster in cluster_df2, find the min p value
        cluster1_id_index = p_list.index(min(p_list)) # which cluster in cluster_df1 is the most correlated to cluster2
        p_cluster_df2.append(min_p)
        corr_cluster_df2.append(corr_list[cluster1_id_index]) # add the correlation from the one with the smallest p value
        
        # add cluster2 as the key and cluster1 as a value to a dictionary as the most correlated clusters
        cluster_pairs[cluster2] = cluster_df1[cluster_column].unique()[cluster1_id_index] 
    # determine correlation with random shuffle
    p_random_mean = []
    p_random_sigma = []
    corr_random_mean = []
    corr_random_sigma = []
    for cluster2 in cluster_df2[cluster_column].unique():
        # generate random permutation to assign membership to nodes in cluster df2 and obtain p values and correlation scores
        p_random, corr_random = permute_cluster_label(expression_meta_df, cluster_df1, cluster_df2, cluster_pairs[cluster2], cluster2, cluster_column)

        p_random_mean.append(p_random.mean())
        corr_random_mean.append(corr_random.mean())
        p_random_sigma.append(p_random.std())
        corr_random_sigma.append(corr_random.std())

    z_p_value = get_z_score(p_cluster_df2, p_random_mean, p_random_sigma)
    z_corr = get_z_score(corr_cluster_df2, corr_random_mean, corr_random_sigma)
    z_score_df = pd.DataFrame({cluster_column:cluster_df2[cluster_column].unique(), 'Z_p_value': z_p_value, 'Z_corr': z_corr})
    return cluster_pairs, z_score_df.sort_values(cluster_column).reset_index(drop = True)


def get_z_score(value, mu, sigma):
    '''calculate z scores with mu: mean and sigma: standard deviation provided'''
    if type(value) == list:
        value = np.array(value)
    if type(mu) == list:
        mu = np.array(mu)
    if type(sigma) == list:
        sigma = np.array(sigma)
    return (value - mu)/sigma

def plot_random_vs_actual_z(cluster_df1, cluster_df2, cluster1, cluster2, cluster_column, network_cluster_stability_df, network_comparison_name, expression_meta_df):
    '''
    After getting network_cluster_stability df from the network_cluster_stability() function, this function plots z score distribution if z score is obtained from random cluster assignment. Red line will show the actual z score from the actual cluster assignment 
    Very small z scores for p values are signs of strong correlation between clusters from 2 networks
    Very small/large z scores for correlation coefficient are signs of strong negative/positive correlation between clusters from 2 networks
    
    '''
    
    p_random, corr_random = permute_cluster_label(expression_meta_df, cluster_df1, cluster_df2, cluster1, cluster2, cluster_column) # generate random p and corr values 100 times
    z_p_value = []
    z_corr = []
    for i in range(100): # take 1 from the 100 shuffles each time and use it as an example to calculate what z score would be if the cluster membership is assigned randomly
        p_pick1 = p_random[i] # take 1 p-value
        corr_pick1 = corr_random[i] # take 1 correlation value
        p_random_rest = np.delete(p_random, i) # the rest of the p-values
        corr_random_rest = np.delete(corr_random, i) # the rest of the correlation value
        z_p_value.append(get_z_score(p_pick1, p_random_rest.mean(), p_random_rest.std())) # calculate z score for p-value
        z_corr.append(get_z_score(corr_pick1, corr_random_rest.mean(), corr_random_rest.std())) # calculate z score for correlation value
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize = (13, 5))
    plt.subplot(1,2,1)
    plt.hist(z_p_value, bins = 25)
    plt.vlines(network_cluster_stability_df[network_cluster_stability_df[cluster_column] == cluster2]['Z_p_value'], 0, 110, color = 'r') # the actual 
    plt.title('Distribution Z_p-value')
    plt.subplot(1,2,2)
    plt.hist(z_corr, bins = 25)
    plt.vlines(network_cluster_stability_df[network_cluster_stability_df[cluster_column] == cluster2]['Z_corr'], 0, 110, color = 'r')
    plt.title('Distribution Z_corr')
    plt.suptitle(f'Z scores {network_comparison_name}: cluster {cluster2}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(Result.getPath(), f'plot_random_vs_actual_z_{str(cluster2)}.png'))
    plt.show()
    plt.close()
    
def gene_phenotype_corr(critical_genes, expression_meta_df, title):
    '''
    Plot correlation heatmap between critical genes and alcohol phenotypes
    '''
    i = 1    
    phenotypes = ['AUDIT', 'Alcohol_intake_gmsperday', 'Total_drinking_yrs']
    for pheno in phenotypes:
        corr_list = []
#         p_list = []
#         corrected_p_list = []
        labels = []
        for gene in critical_genes:
            if gene in expression_meta_df.columns:
                sub = expression_meta_df[[gene, pheno]]
                sub = sub.dropna()
                corr_list.append(pearsonr(sub[gene], sub[pheno])[0])
#                 p_list.append(pearsonr(sub[gene], sub[pheno])[1])
#         corrected_p_list = multipletests(p_list, method ='fdr_bh')[1] # correct for multiple tests
        if i == 1:
            genes_corr = pd.DataFrame({pheno: corr_list})
#             genes_pvalue = pd.DataFrame({pheno: corrected_p_list})
            i += 1

        else:
            genes_corr[pheno] = corr_list
#             genes_pvalue[pheno] = corrected_p_list
    genes_corr.index = critical_genes
    sort_corr = genes_corr.reindex(genes_corr.mean(axis = 1).sort_values().index) 
    # sort index by index mean (not abs so the pos and neg correlation are divergent)
#     genes_pvalue.index = critical_genes
    plt.rcParams.update({'font.size':14})
    plt.figure(figsize = (6, 11))
#     gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])  # set the subplot width ratio
#     # first subplot to show the correlation heatmap
#     ax0 = plt.subplot(gs[0])
# #     sns.heatmap(genes_corr.sort_index(), cmap='RdBu_r', annot = True,
# #                 annot_kws = {'fontsize':12}, vmin=-1, vmax=1, xticklabels = phenotypes) 

#     plt.xticks(rotation = 45, ha = 'right')
#     plt.ylabel('Gene symbol')
#     plt.title('Trait gene correlation')
#     # second subplot to show count of significant traits in each cluster. "Significant" here means adj p value < 0.2
#     ax1 = plt.subplot(gs[1])
#     genes_pvalue = genes_pvalue
#     sig_count = (genes_pvalue < 0.2).sum(axis = 1) # count num of traits with p-adj < 0.2 in each cluster
#     sig_count = sig_count.sort_index(ascending = False)
#     plt.barh(sig_count.index, sig_count.values) # horizontal bar plot
#     plt.xlim(0,9) # there are 9 traits here so set the scale to between 0 and 9. change it if the # traits change
#     plt.ylabel('Gene symbol')
#     plt.xlabel('Trait count')
#     plt.title('Number of correlated traits')
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()
#     plt.subplots_adjust(wspace = 1)

    plt.title(title)
    sns.heatmap(sort_corr, cmap='RdBu_r', vmin = -0.5, vmax=0.5, xticklabels = phenotypes, yticklabels = True)
    plt.xticks(rotation = 45, ha = 'right')
    plt.ylabel('Gene Symbol')
    return genes_corr
    




    
    
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
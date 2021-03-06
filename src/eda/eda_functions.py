import sys
sys.path.append('../../src')
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
from matplotlib import gridspec
from scipy.stats import f_oneway, ttest_ind
from sknetwork.clustering import Louvain
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
from .process_phenotype import *
from preproc.result import Result

def plot_gene_cnt_each_cluster(cluster_dfs, network_names):
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
        count = cluster_df['cluster_id'].value_counts().sort_index()
        plt.bar(count.index, count.values)
#         if type(count.index[0]) == np.int64:
#             plt.xticks(list(np.arange(0,len(count.index),1)))        
        plt.ylabel('# genes')
        plt.xlabel('Cluster id')
        plt.title(network_names[i])
        plt.subplots_adjust(wspace = 0.3)
    plt.tight_layout()    
    plt.savefig(os.path.join(Result.getPath(), "plot_gene_cnt_each_cluster.png"))
    plt.show()
    plt.close()

def plot_gene_cnt_each_cluster_v2(cluster_df, network_name):
    '''
    bar graphs to show # genes in each cluster
    cluster_dfs: a list of cluster dfs with id and cluster assignment
    cluster_column: cluster type, louvain or k means
    network_names: names to show in the subplot titles
    '''
    plt.rcParams.update({'font.size': 18})
    count = cluster_df['cluster_id'].value_counts().sort_index()
#     plt.figure(figsize = (len(count.index)/2.5,4))
    plt.figure(figsize = (8,6))
    plt.bar(count.index, count.values)
    if type(count.index[0]) == np.int64:
        plt.xticks(list(np.arange(0,len(count.index),1)))
    plt.ylabel('# genes')
    plt.xlabel('Cluster id')
    plt.title(network_name)
    plt.tight_layout()
    plt.savefig(os.path.join(Result.getPath(), f"plot_gene_cnt_each_cluster_{network_name}.png"))
    plt.show()
    plt.close()

def run_kmeans(embedding_df, n_clusters):
    '''Run k means on embedding df'''
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(embedding_df)
    k_mean_df = pd.DataFrame({'id':embedding_df.index, 'cluster_id':kmeans})
    return k_mean_df

def run_louvain(adjacency_df, resolution = 1, n_aggregations = -1):
    # louvain communities
    louvain = Louvain(modularity = 'Newman', resolution = resolution, n_aggregations  = n_aggregations)
    labels = louvain.fit_transform(adjacency_df.values) # using networkx community requires converting the df to G first and the original network takes very long but this method can work on df 
    louvain_df = pd.DataFrame({'id':adjacency_df.index, 'cluster_id':labels})
    return louvain_df

def run_louvain2(adjacency_np, ajacency_idx, resolution = 1, n_aggregations = -1):
    # louvain communities
    louvain = Louvain(modularity = 'Newman', resolution = resolution, n_aggregations  = n_aggregations)
    labels = louvain.fit_transform(adjacency_np) # using networkx community requires converting the df to G first and the original network takes very long but this method can work on df 
    del louvain
    louvain_df = pd.DataFrame({'id':ajacency_idx, 'cluster_id':labels})
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

def cluster_jaccard_v2(cluster_df1, cluster_df2, comparison_names, 
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

    for c1 in cluster_df1['cluster_id'].unique():
        for c2 in cluster_df2['cluster_id'].unique():
            sub1 = cluster_df1[cluster_df1['cluster_id'] == c1].index
            sub2 = cluster_df2[cluster_df2['cluster_id'] == c2].index
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

def plot_sig_perc(cluster_df, network_name, expression_meta_df, output_sig_df = False):
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
        clusters = cluster_df['cluster_id'].unique()
        for cluster in clusters:
            sig_gene_perc.append(get_module_sig_gene_perc(subset, cluster_df, 'cluster_id', cluster, traits[i]))
        if i == 0:
            cluster_sig_perc = pd.DataFrame({traits[i]: sig_gene_perc})
        else:
            cluster_sig_perc[traits[i]] = sig_gene_perc
    cluster_sig_perc.index = clusters    
    cluster_sig_perc = cluster_sig_perc.sort_index(ascending = False)
    fig = plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.5)
    sns.set_style('white')
#     gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])  # set the subplot width ratio
#     # first subplot to show the correlation heatmap
#     ax0 = plt.subplot(gs[0])
    sns.heatmap(cluster_sig_perc, cmap='Reds',
                vmin=0, vmax=15) 
    plt.xticks(rotation = 45, ha = 'right')
    plt.ylabel('cluster id')
#     plt.title('% significant genes by cluster')
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

def cluster_phenotype_corr(cluster_df, network_name, expression_meta_df, output_corr_df = False):
    '''
    Plot correlation heatmap between modules/clusters and alcohol phenotypes
    '''
    clusters = cluster_df['cluster_id'].unique()
    i = 1
    for cluster in clusters:
        cluster_genes = cluster_df[cluster_df['cluster_id'] == cluster]['id'].tolist()
        cluster_expression = expression_meta_df[cluster_genes].apply(pd.to_numeric)
        pca = PCA(n_components=1)
        pca_cluster_expression = pca.fit_transform(cluster_expression)
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
    sns.set(font_scale=1.5)
    sns.set_style('white')
    clusters_corr.columns = eigen_n_features.columns[1:]
#     gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # set the subplot width ratio
#     # first subplot to show the correlation heatmap
#     ax0 = plt.subplot(gs[0])
    sns.heatmap(clusters_corr, cmap='RdBu_r', annot = True,
                annot_kws = {'fontsize':12}, vmin=-1, vmax=1, xticklabels = eigen_n_features.columns[1:]) 
    plt.xticks(rotation = 45, ha = 'right')
    plt.yticks(rotation = 0)
    plt.ylabel('cluster id')
    plt.title(f'Trait cluster correlation for {network_name}')
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

def cluster_nmi_v3(cluster_df1, cluster_df2):
    '''NMI to compare communities from the whole netowrk and the subnetwork or clusters from different network embeddings'''
    sub1_plus_sub2 = pd.merge(cluster_df1, cluster_df2, left_on = 'id', right_on = 'id', how = 'outer', suffixes = ('_1','_2'))
    max_cluster_num = max(sub1_plus_sub2[['cluster_id_1', 'cluster_id_2']].max())
    sub1_plus_sub2['cluster_id_2'].fillna(max_cluster_num+1, inplace = True) # for the nodes that were cut out, give them a new community number
    return round(nmi(sub1_plus_sub2['cluster_id_1'], sub1_plus_sub2['cluster_id_2']),3)
    
def cluster_DE_perc(cluster_df, network_name, deseq):
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
    for cluster in cluster_df['cluster_id'].unique():
        cluster_genes = cluster_df[cluster_df['cluster_id'] == cluster].id
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
        labels = []
        for gene in critical_genes:
            if gene in expression_meta_df.columns:
                sub = expression_meta_df[[gene, pheno]]
                sub = sub.dropna()
                corr_list.append(pearsonr(sub[gene], sub[pheno])[0])
#                 p_list.append(pearsonr(sub[gene], sub[pheno])[1])
        if i == 1:
            genes_corr = pd.DataFrame({pheno: corr_list})
#             corr_p = pd.DataFrame({pheno: p_list})
            i += 1
        else:
            genes_corr[pheno] = corr_list
#             corr_p[pheno] = p_list
    genes_corr.index = critical_genes
#     corr_p.index = critical_genes
    sort_corr = genes_corr.reindex(genes_corr.mean(axis = 1).sort_values().index) 
#     sort_p = corr_p.reindex(genes_corr.mean(axis = 1).sort_values().index)
    plt.rcParams.update({'font.size':14})
    plt.figure(figsize = (6, 11))
    plt.title(title, fontsize = 26)
    sns.heatmap(sort_corr, cmap='RdBu_r', vmin = -0.5, vmax=0.5, xticklabels = phenotypes, yticklabels = True)
    plt.xticks(rotation = 45, ha = 'right', fontsize = 26)
    plt.ylabel('Gene symbol', fontsize = 26)
    plt.savefig(os.path.join(Result.getPath(), f'gene_phenotype_corr_for_{title}.png'), bbox_inches='tight')
    plt.show()
    plt.close()
    return genes_corr

def plot_corr_kde(corr_df_list, corr_names, plotname):
    new_corr_df_list = []
    for corr_df, name in zip(corr_df_list, corr_names):
        corr_copy = np.abs(corr_df.copy())
        corr_copy['sample'] = name
        new_corr_df_list.append(corr_copy)   
    p_values = []
    for col in ['AUDIT', 'Alcohol_intake_gmsperday', 'Total_drinking_yrs']:
        if len(new_corr_df_list) > 2:
            round_p = round(f_oneway(*[corr_df[col] for corr_df in new_corr_df_list])[1], 3)
            p_values.append(round_p)
        else:
            corr1 = new_corr_df_list[0][col]
            corr2 = new_corr_df_list[1][col]
            round_p = round(ttest_ind(corr1, corr2)[1], 3)
            p_values.append(round_p)
    joined_corr = pd.concat(new_corr_df_list)                
    melt_df = pd.melt(joined_corr, id_vars=['sample'])
    sns.set(font_scale=1.5)
    sns.set_style('white')
    g = sns.FacetGrid(melt_df, col='variable', hue = 'sample', height = 4, aspect = 1.2)
    g.map(sns.kdeplot, 'value')  
    g.set_axis_labels(x_var = 'Absolute correlation coefficient')
    g.add_legend()
    g._legend.set_title('')
    plt.setp(g._legend.get_title(), fontsize=20)
    axes = g.axes.flatten()
    for ax, p in zip(axes, p_values):
        title = ax.get_title()
        new_title = title.replace('variable = ', '')
        new_title = f'{new_title} (p={p})' if p > 0 else f'{new_title} (p < 0.001)'
        ax.set_title(new_title)
    plt.savefig(os.path.join(Result.getPath(), f'alcohol trait correlation {plotname}.png'))
    plt.show()
    plt.close()
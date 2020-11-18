from node2vec import Node2Vec
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


def network_embedding(graph, walk_length, num_walks, window, output_dir = None, name_spec = ''):
    '''
    graph: networkx G for a network
    walk_length, num_walks, window: parameters for embedding
    name_spec: any additional info as str to add for saving the embedding df
    '''
    node2vec = Node2Vec(graph, dimensions=64, walk_length=walk_length, num_walks=num_walks)
    model = node2vec.fit(window = window, min_count=1, workers = 4)
    emb_df = pd.DataFrame(np.asarray(model.wv.vectors), index = graph.nodes)
    if output_dir:
        emb_df.to_csv(f'{output_dir}/embedded_len{walk_length}_walk{num_walks}_{name_spec}.csv')
        print('embedding data saved')
    return emb_df

def run_kmeans(embedding_df, n_clusters):
    '''Run k means on embedding df'''
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(embedding_df)
    k_mean_df = pd.DataFrame({'id':embedding_df.index, 'kmean_label':kmeans})
    return k_mean_df

def jaccard_similarity(list1, list2):
    intersection = len(set(list1).intersection(list2))
    union = len(set(list1)) + len(set(list2)) - intersection
    return intersection / union


def cluster_jaccard(cluster_df1, cluster_df2, cluster_column, comparison_names, top=None):
    '''
    plot jaccard pairwise comparison on the communities in 2 networks or the kmeans clusters in 2 network embeddings
    title: main title for the two subplots
    cluster_column: the column name of the cluster labels
    comparison_names: names of the groups in comparison
    top: top n comparison to show in the boxplot since it could be misleadingly small if we include all jaccard scores
    # we're only interested in the modules that have majority of the matching nodes between 2 networks
    '''
    c1_list = []
    c2_list = []
    j_list = []
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

    fig = plt.figure(figsize=(8, 5))
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
        sns.boxplot(x=None, y=sorted_jac_values[:top])
    else:
        sns.boxplot(x=None, y=all_jac_values)
    plt.ylim(0, 1)
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

def plot_sig_perc(expression_meta_df, cluster_df, cluster_column, trait, network_name):
    sig_gene_perc = []
    clusters = cluster_df[cluster_column].unique()
    for cluster in clusters:
        sig_gene_perc.append(get_module_sig_gene_perc(expression_meta_df, cluster_df, cluster_column, cluster, trait))

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
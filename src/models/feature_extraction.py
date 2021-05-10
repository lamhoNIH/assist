import pandas as pd
import numpy as np
from itertools import combinations
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as ed
from collections import Counter
from functools import reduce
from preproc.result import Result


def get_min_max_center(coef):
    '''A function to get min, max and avg used for heatmap in plot_feature_importances()'''
    min_value = coef.min()
    max_value = coef.max()
    center = np.mean([min_value, max_value])
    return min_value, max_value, center

def plot_feature_importances(model_weights, top_n_coef = 0.2, print_num_dim = True, plot_heatmap = False, return_top_dim = False):
    '''
    A function to show feature importances in each model
    and can return feature importance and the top dim from each model
    '''
    models = ['LR']*3 + ['RF']*3 + ['XGB']*3 
    if plot_heatmap == True:
        sns.set(font_scale=1.5)
        sns.set_style('white')
#         plt.figure(figsize = (0.5, 20))
        i = 0
        for coef in model_weights:
            plt.figure(figsize = (1,9))
            min_v, max_v, center = get_min_max_center(coef)
#             plt.subplot(3, 3,i+1)
            sns.heatmap(coef.reshape(64, 1), center = center, vmin = min_v, vmax = max_v, cmap = 'Reds', xticklabels = [])
            plt.title(models[i], fontsize=24)
            plt.yticks(rotation = 0)
            plt.show()
            plt.close()
#             plt.subplots_adjust(wspace = 2)
            i += 1
    if print_num_dim == True and return_top_dim == True:
        top_dim_list = list(map(get_top_dim, model_weights, models, [True]*len(model_weights),
                                [top_n_coef]*len(model_weights), [True]*len(model_weights)))
        return top_dim_list
    elif print_num_dim == False and return_top_dim == True:
        top_dim_list = list(map(get_top_dim, model_weights, models, [False]*len(model_weights),
                                [top_n_coef]*len(model_weights), [True]*len(model_weights)))
        return top_dim_list
    
def get_top_dim(coef, model_name, print_num_dim = True, top_n_coef = 0.2, return_top_dim = False, dimensions = 64):
    '''
    Get the top features used for each ML and return as a list along with id and abs_log2FC for the get_pairwise_distances()
    top_n_coef: 0.2 means extract features up to 20% importance
    '''
    for i in range(1, dimensions):
        if np.sum(coef[coef.argsort()[-i:]]) > top_n_coef:
            num_dim = i
            break 
    top_dim = [str(num) for num in coef.argsort()[-num_dim:]]
    if print_num_dim == True:
        print(f'Number of dim for {model_name}:', len(top_dim))
    
    if return_top_dim == True:
        top_dim += ['id', 'abs_log2FC']
        return top_dim

def jaccard_similarity(list1, list2):
    intersection = len(set(list1).intersection(list2))
    union = len(set(list1)) + len(set(list2)) - intersection
    return intersection / union

def jaccard_average(top_dim_list, title):
    new_top_dim_list = [dim_list[:-2] for dim_list in top_dim_list]
    jac_average = []
    for i in range(0, len(new_top_dim_list), 3): # each model was repeated 3 times 
        jac_list = []
        for dim_list1, dim_list2 in combinations(new_top_dim_list[i:i+3],2): # compare 2 out of 3 with all combinations
            jac_list.append(jaccard_similarity(dim_list1, dim_list2))
        jac_average.append(np.mean(jac_list))
    plt.rcParams.update({'font.size': 18})
    plt.bar(['lr', 'rf', 'xgb'], jac_average)
    plt.ylim(0, 1)
    plt.ylabel('jaccard similarity')
    plt.title(title)
    plt.savefig(os.path.join(Result.getPath(), f'jaccard_average_{title}.png'), bbox_inches='tight')
    plt.show()
    plt.close()
    
def plot_random_feature_importance(feature_importance_list, top_dim_list, subnetwork_name):
    models = ['LR', 'RF', 'XGB']
    plt.figure(figsize = (15, 3))
    plt.rcParams.update({'font.size': 18})
    for j in range(0, 9, 3):
        l = int(j/3)
        top_dim = top_dim_list[j][:-2]
        top_dim = [int(d) for d in top_dim]
        actual_importance = sum(feature_importance_list[j][top_dim])
        random_dim_importance = []

        num_dim = len(top_dim)
        for i in range(500):
            random_dim_importance.append(sum(np.random.choice(feature_importance_list[j], num_dim, replace = False)))

        plt.subplot(1,3,l+1)
        plt.hist(random_dim_importance)
        plt.axvline(actual_importance, 0, 10, color = 'r')
        plt.title(models[l] + ' '+ subnetwork_name)
        plt.xlabel('importance sum')
        plt.ylabel('events')
        plt.subplots_adjust(wspace = 0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(Result.getPath(), f'plot_random_feature_importance_{models[l]}_{subnetwork_name}.png'))
    plt.show()
    plt.close()

def get_pairwise_distances(processed_emb_df):
    '''Determine pairwise euclidean distance between each data point'''
    pairwise_distances = pd.DataFrame(ed(processed_emb_df.iloc[:, :-2]))
    pairwise_distances.columns = processed_emb_df['id']
    pairwise_distances.index = processed_emb_df['id']
    pairwise_distances['abs_log2FC'] = processed_emb_df['abs_log2FC'].tolist()
    pairwise_distances.sort_values('abs_log2FC', ascending = False, inplace=True)
    return pairwise_distances


def get_critical_genes(pairwise_distance_df, cutoff, max_dist = 0.55):
    '''
    Find critical genes that are close to impact genes
    critical_gene_dict: # impact genes a critical gene is close to
    gene_pair_dict: pair the impact gene with their critical genes (based on distance) in a dictionary
    pairwise_distance_df: pairwise distance between the genes and sorted with abs_log2FC from high to low
    return critical_gene_dict: critical gene as keys, number of impact genes it's close to as the values
    gene_pair_dict: impact genes as keys and their corresponding critical genes as values
    '''
    critical_gene_list = []
    gene_pair_dict = {}
    size = len(pairwise_distance_df[pairwise_distance_df.abs_log2FC > cutoff]) # cutoff of abs_log2FC > 0.2 as impact gene
    for i in range(size):
        subset_distance = pairwise_distance_df.iloc[i,:-2].sort_values()
        key = subset_distance[subset_distance.between(1e-6,max_dist)].reset_index().columns[1]
        values = subset_distance[subset_distance.between(1e-6,max_dist)].reset_index()['id'].tolist() # values are the list of close genes, aka potential critical genes
        gene_pair_dict[key] = values
        critical_gene_list += list(subset_distance[subset_distance.between(1e-6,max_dist)].index)
    critical_gene_dict = Counter(critical_gene_list)
    critical_gene_dict = sorted(critical_gene_dict.items(), key=lambda x: x[1], reverse=True)
    return critical_gene_dict, gene_pair_dict

def top_critical_genes(critical_gene_list, min_gene):
    return [i[0] for i in critical_gene_list[0] if i[1] >= min_gene]

def calculate_distance_stats(distance_df_list):
    '''
    A function to determine smallest distance mean and largest distance mean for each gene identified in the importance dimensions
    This function will provide a sense of what the distance cutoff should be for a gene to be considered as "close" to an impact gene
    '''
    distance_df_joined = pd.concat(distance_df_list)
    max_mean = np.max(distance_df_joined.mean())
    min_mean = np.min(distance_df_joined.mean())
    print('Max mean:', max_mean, 'Min mean:', min_mean)

def get_max_dist(distance_dfs, ratio = 0.7, max_dist_ratio = 6*1e-5):
    '''
    A method to determine max_dist for critical gene identification
    '''
    max_dist_list = []
    for distance in distance_dfs:
        distance_np = distance.iloc[:,:int(len(distance)*0.4)].to_numpy() # use only 40% of the df
        flatten_distance = distance_np.flatten()
        del distance_np
        # max_dist = np.sort(flatten_distance, axis = 0)[int(len(flatten_distance)*6*1e-6)] #8*1e-5 was determined by manual testing
        max_dist = np.sort(flatten_distance, axis=0)[int(len(flatten_distance) * max_dist_ratio)]
        i = 1
        while len(flatten_distance[flatten_distance < max_dist]) < len(distance)*ratio:
            i += 1
            max_dist = np.sort(flatten_distance, axis=0)[int(len(flatten_distance) * i * max_dist_ratio)]
        max_dist_list.append(max_dist)

    return max_dist_list

def get_critical_gene_sets(processed_emb_df, top_dim_list, deseq, ratio = 0.7, max_dist_ratio = 6*1e-5):
    '''
    Input: processed embedding df used for ML and top_dim_list (set of 9 for 3 models x 3 repeats)
    ratio is what % of the genes need to be less than the max_dist for critical gene identification.
    small ratio for fewer critical genes and large ratio for more critical genes
    Output: 9 sets of critical genes for 3 models x 3 repeats
    '''
    if 'abs_log2FC' not in deseq.columns:
        deseq['abs_log2FC'] = abs(deseq['log2FoldChange'])
    process_emb_df_subset = [processed_emb_df[top_dim_list[i]] for i in range(9)]
    distance_dfs = []
    for process_emb_df in process_emb_df_subset:
        distance_dfs.append(get_pairwise_distances(process_emb_df))
    # return distance_dfs
    cutoff = deseq['abs_log2FC'].sort_values(ascending = False).reset_index(drop = True)[int(len(deseq) * 0.01)]
    max_dist_list = get_max_dist(distance_dfs, ratio = ratio, max_dist_ratio = max_dist_ratio)
#     calculate_distance_stats(distance_dfs)
    critical_gene_sets = list(map(get_critical_genes, distance_dfs, [cutoff]*len(distance_dfs), max_dist_list))
    return critical_gene_sets


def get_critical_gene_df(critical_gene_set, network_name, output_path):
    '''
    Supply 9 sets of critical genes for 3 model x 3 repeats 
    Return a critical gene df with count in each model and each repeat 
    with the sum of how many the critical gene shows up near an impact gene
    '''
    models = ['LR','RF','XGB']
    critical_gene_dfs = []
    for j in range(0, 9, 3):
        l = int(j/3)
        for i in range(3):
            temp = pd.DataFrame(critical_gene_set[i+j][0], columns = ['gene', f'{models[l]}_repeat{i+1}'])
            critical_gene_dfs.append(temp)
        critical_gene_dfs_merged = reduce(lambda left,right:pd.merge(
            left,right,on=['gene'],how='outer'), critical_gene_dfs)
        critical_gene_dfs_merged.fillna(0, inplace = True)

    critical_gene_dfs_merged['near_impact_cnt'] = critical_gene_dfs_merged.sum(axis = 1)
    critical_gene_dfs_merged = critical_gene_dfs_merged.sort_values(
        'near_impact_cnt', ascending = False).reset_index(drop = True)
    critical_gene_dfs_merged.to_csv(output_path, index = 0)
    return critical_gene_dfs_merged

def plot_nearby_impact_num(critical_gene_df, emb_name, top = 10):
    '''Plot count of nearby impact genes for each set of critical gene df'''
    critical_df = critical_gene_df[['gene', 'near_impact_cnt']].loc[:top,]
    critical_df.sort_values('near_impact_cnt', inplace = True)
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['axes.titlepad'] = 15
    plt.barh(critical_df['gene'], critical_df['near_impact_cnt'])
    plt.xlabel('Number of nearby impact genes')
    plt.ylabel('Critical gene ID')
    plt.title(emb_name)
#     plt.tight_layout()
    plt.savefig(os.path.join(Result.getPath(), f'plot_nearby_impact_num_{emb_name}.png'), bbox_inches='tight')
    plt.show()
    plt.close()
    return critical_df['gene']

def jaccard_critical_genes(critical_gene_df, network_name):
    '''
    jaccard similarity between top 10 critical genes identified by each model
    '''
    critical_gene_df['lr'] = critical_gene_df['LR_repeat1'] + critical_gene_df['LR_repeat2'] + critical_gene_df['LR_repeat3']
    critical_gene_df['rf'] = critical_gene_df['RF_repeat1'] + critical_gene_df['RF_repeat2'] + critical_gene_df['RF_repeat3']
    critical_gene_df['xgb'] = critical_gene_df['XGB_repeat1'] + critical_gene_df['XGB_repeat2'] + critical_gene_df['XGB_repeat3']
    cols_to_permute = ['lr', 'rf', 'xgb']
    jaccard_list = []
    model_names = []
    for col1, col2 in combinations(cols_to_permute, 2):
        top10_1 = critical_gene_df.sort_values(col1, ascending = False)['gene'][:10]
        top10_2 = critical_gene_df.sort_values(col2, ascending = False)['gene'][:10]
        jaccard_list.append(jaccard_similarity(top10_1, top10_2))
        model_names.append(f'{col1} vs {col2}')
    plt.rcParams.update({'font.size': 18})    
    plt.bar(model_names, jaccard_list)
    plt.title(network_name)
    plt.ylabel('jaccard similarity')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(Result.getPath(), f'jaccard_critical_genes_{network_name}.png'))
    plt.show()
    plt.close()
    
    gene_sets = [set(critical_gene_df[['gene',col]].sort_values(col, ascending = False)[:10]['gene']) for col in cols_to_permute]
    intersect_genes = set.intersection(*gene_sets)
    return intersect_genes
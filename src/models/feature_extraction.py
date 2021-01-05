import pandas as pd
import numpy as np
from sys import platform
from itertools import combinations
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as ed
from collections import Counter
from functools import reduce

def get_min_max_center(coef):
    '''A function to get min, max and avg used for heatmap in plot_feature_importances()'''
    min_value = coef.min()
    max_value = coef.max()
    center = np.mean([min_value, max_value])
    return min_value, max_value, center

def get_important_features(model):
    '''Get feature importances from models'''
    if type(model).__name__ == 'LogisticRegression':
        coef = model.coef_[0]
        coef = np.abs(coef) # convert coef to positive values only
        coef /= np.sum(coef) # convert coef to % importance
    else:
        coef = model.feature_importances_
    return coef

def plot_feature_importances(model_path, top_n_coef = 0.2, print_num_dim = True, plot_heatmap = False, return_top_dim = False):
    '''
    A function to show feature importances in each model
    and can return feature importance and the top dim from each model
    '''
    model_files = os.listdir(model_path)
    model_list = []
    for file in model_files:
        with open(model_path + '/' + file, 'rb') as model:
            model_list.append(pickle.load(model))
    models_feature_importances = list(map(get_important_features, model_list))
    if plot_heatmap == True:
        plt.figure(figsize = (7, 23))
        i = 0
        for coef in models_feature_importances:
            min_v, max_v, center = get_min_max_center(coef)
            plt.subplot(3, 3,i+1)
            sns.heatmap(coef.reshape(64, 1), center = center, vmin = min_v, vmax = max_v)
            plt.title(model_files[i])
            plt.subplots_adjust(wspace = 2)
            i += 1
    if print_num_dim == True and return_top_dim == True:
        top_dim_list = list(map(get_top_dim, models_feature_importances, model_files, [True]*len(models_feature_importances),
                                [top_n_coef]*len(models_feature_importances), [True]*len(models_feature_importances)))
        return models_feature_importances, top_dim_list
    elif print_num_dim == False and return_top_dim == True:
        top_dim_list = list(map(get_top_dim, models_feature_importances, model_files, [False]*len(models_feature_importances),
                                [top_n_coef]*len(models_feature_importances), [True]*len(models_feature_importances)))
        return models_feature_importances, top_dim_list
    elif print_num_dim == True and return_top_dim == False:
        top_dim_list = list(map(get_top_dim, models_feature_importances, model_files, [True]*len(models_feature_importances),
                                [top_n_coef]*len(models_feature_importances), [False]*len(models_feature_importances)))
        return models_feature_importances
    
def get_top_dim(coef, model_name, print_num_dim = True, top_n_coef = 0.2, return_top_dim = False):
    '''
    Get the top features used for each ML and return as a list along with id and abs_log2FC for the get_pairwise_distances()
    top_n_coef: 0.2 means extract features up to 20% importance
    '''
    for i in range(1,64):
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
    plt.bar(['lr', 'rf', 'xgb'], jac_average)
    plt.ylim(0, 1)
    plt.ylabel('jaccard similarity')
    plt.title(title)
    plt.show()
    plt.close()
    
def plot_random_feature_importance(feature_importance_list, top_dim_list, subnetwork_name):
    models = ['LR', 'RF', 'XGB']
    plt.figure(figsize = (18, 3))
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
        plt.subplots_adjust(wspace = 0.2);
        
        
def get_pairwise_distances(processed_emb_df):
    '''Determine pairwise euclidean distance between each data point'''
    pairwise_distances = pd.DataFrame(ed(processed_emb_df.iloc[:, :-2]))
    pairwise_distances.columns = processed_emb_df['id']
    pairwise_distances.index = processed_emb_df['id']
    pairwise_distances['abs_log2FC'] = processed_emb_df['abs_log2FC'].tolist()
    pairwise_distances.sort_values('abs_log2FC', ascending = False, inplace=True)
    return pairwise_distances

def get_critical_genes(pairwise_distance_df, max_dist = 0.55):
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
    size = len(pairwise_distance_df[pairwise_distance_df.abs_log2FC > 0.2]) # cutoff of abs_log2FC > 0.2 as impact gene
    for i in range(size):
        subset_distance = pairwise_distance_df.iloc[i,:-2].sort_values()
        key = subset_distance[subset_distance.between(0.01,max_dist)].reset_index().columns[1] # Euclidean distance < 0.55 as "close", key is an impact gene when 20% important features were used. Increase the size when more features are used 
        values = subset_distance[subset_distance.between(0.01,max_dist)].reset_index()['id'].tolist() # values are the list of close genes, aka potential critical genes
        gene_pair_dict[key] = values
        critical_gene_list += list(subset_distance[subset_distance.between(0.01,max_dist)].index)
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
    
def get_critical_gene_sets(processed_emb_df, top_dim_list, max_dist = 0.55):
    '''
    Input: processed embedding df used for ML and top_dim_list (set of 9 for 3 models x 3 repeats)
    Output: 9 sets of critical genes for 3 models x 3 repeats
    '''
    process_emb_df_subset = [processed_emb_df[top_dim_list[i]] for i in range(9)]
    distance_dfs = list(map(get_pairwise_distances, process_emb_df_subset))
    calculate_distance_stats(distance_dfs)
    critical_gene_sets = list(map(get_critical_genes, distance_dfs, [max_dist]*len(distance_dfs)))
    return critical_gene_sets


def get_critical_gene_df(critical_gene_set):
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
            temp = pd.DataFrame(critical_gene_set[i+j][0])
            temp.columns = ['gene', f'{models[l]}_repeat{i+1}']
            critical_gene_dfs.append(temp)
        critical_gene_dfs_merged = reduce(lambda left,right:pd.merge(
            left,right,on=['gene'],how='outer'), critical_gene_dfs)
        critical_gene_dfs_merged.fillna(0, inplace = True)

    critical_gene_dfs_merged['near_impact_cnt'] = critical_gene_dfs_merged.sum(axis = 1)
    critical_gene_dfs_merged = critical_gene_dfs_merged.sort_values(
        'near_impact_cnt', ascending = False).reset_index(drop = True)
    return critical_gene_dfs_merged


def plot_nearby_impact_num(critical_gene_df, emb_name, top = 10):
    '''Plot count of nearby impact genes for each set of critical gene df'''
    critical_df = critical_gene_df[['gene', 'near_impact_cnt']].loc[:10,]
    critical_df.sort_values('near_impact_cnt', inplace = True)
    plt.barh(critical_df['gene'], critical_df['near_impact_cnt'])
    plt.xlabel('Number of nearby impact genes')
    plt.ylabel('Critical gene ID')
    plt.title(emb_name)
    plt.show()
    plt.close()
    return critical_df['gene']
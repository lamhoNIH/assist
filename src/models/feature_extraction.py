import pandas as pd
import numpy as np
from sys import platform
from ..preproc.deseq_data import deseq
from itertools import combinations


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

def plot_feature_importances(model_path, print_num_dim = True, plot_heatmap = False, return_top_dim = False):
    '''A function to show feature importances in each model'''
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
                                [0.2]*len(models_feature_importances), [True]*len(models_feature_importances)))
        return models_feature_importances, top_dim_list
    elif print_num_dim == False and return_top_dim == True:
        top_dim_list = list(map(get_top_dim, models_feature_importances, model_files, [False]*len(models_feature_importances),
                                [0.2]*len(models_feature_importances), [True]*len(models_feature_importances)))
        return models_feature_importances, top_dim_list
    else:
        top_dim_list = list(map(get_top_dim, models_feature_importances, model_files))
        return models_feature_importances
    
def get_top_dim(coef, model_name, print_num_dim = True, top_n_coef = 0.2, return_top_dim = False):
    '''
    Get the top features used for each ML and return as a list along with id and abs_log2FC for the get_pairwise_distances()
    top_n_coef: 0.2 means extract features up to 20% importance
    '''
    for i in range(64):
        if np.sum(coef[coef.argsort()[-i:]]) > top_n_coef:
            num_dim = i
            break 
    top_dim = [str(num) for num in coef.argsort()[-num_dim:]]
    if print_num_dim == True:
        print(f'Number of dim for {model_name}:', len(top_dim))
    
    if return_top_dim == True:
        top_dim += ['id', 'abs_log2FC']
        return top_dim
    
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
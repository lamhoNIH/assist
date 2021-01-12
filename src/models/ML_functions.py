import pandas as pd
import numpy as np
from ..preproc.deseq_data import DESeqData
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
import seaborn as sns

def process_emb_for_ML(embedding_df, deseq = DESeqData.get_deseq()):
    embedding_labeled_df = pd.merge(embedding_df, deseq, left_index = True, right_on = 'id')
    embedding_labeled_df['impact'] = 1
    embedding_labeled_df.loc[embedding_labeled_df['log2FoldChange'].between(-0.1, 0.1), 'impact'] = 0
    return embedding_labeled_df

def run_test_harness_ml(embedding_data, output_dir, description,
           feature_cols_to_use, models_to_test, repeat=6, feature_cols_to_normalize=None,
           sparse_cols_to_use=None, pred_col='impact'):
    '''Use test-harness to run ML to predict impact/non-impact genes with embedding features'''
    th_path = output_dir
    th = TestHarness(output_location = th_path)
    num_sample = embedding_data['impact'].value_counts().min()
    for i in range(repeat):
        emb_subset = embedding_data.groupby('impact').sample(num_sample).reset_index() # subset to have equal samples in the two classes
        train_df, test_df = train_test_split(emb_subset, test_size = 0.2)
        normalize = False
        if feature_cols_to_normalize:
            normalize = True

        for model in models_to_test:
            th.run_custom(function_that_returns_TH_model = model,
                          dict_of_function_parameters={}, training_data = train_df,
                          testing_data = test_df, description = description,
                          target_cols=pred_col, feature_cols_to_use = feature_cols_to_use,
                          index_cols = ['index_col'], normalize = normalize,
                          feature_cols_to_normalize = feature_cols_to_normalize, feature_extraction = False,
                          sparse_cols_to_use=sparse_cols_to_use, predict_untested_data=False)
            
def plot_ML_results(th_path, description_list, output_dir = None):
    '''Plot test-harness ML results'''
    model_dict = {'random_forest_classification': 'Random Forest',
                  'gradient_boosted_tree': 'Gradient Boosted Tree',
                  'logistic_classifier': 'Logistic Regression'}
    sns.set(rc={'figure.figsize': (6, 4)}, font_scale=1.5)
    sns.set_style("white")
    for description in description_list:
        leaderboard_df = query_leaderboard(query={'Description':description},
                               th_output_location=th_path, loo=False, classification=True)
        leaderboard_df['Model Name'] = leaderboard_df['Model Name'].map(model_dict)
        leaderboard_df['Accuracy'] = 100*leaderboard_df['Accuracy']
        agg_acc = leaderboard_df.groupby('Model Name')['Accuracy'].mean()
        print('best accuracy', round(agg_acc.max()))
        ax = sns.boxplot(x = 'Model Name', y = 'Accuracy', data = leaderboard_df)
        ax.set(ylim = (0,100))
        plt.title(description)
        plt.xlabel('')
        plt.axhline(50, color = 'r')
        plt.xticks(rotation = 45, ha = 'right')
        plt.show()
        plt.close()
        
        
def run_ml(processed_embedding, emb_name, print_accuracy = False, output_dir = None):
    '''Run ML using sklearn with LR, RF and XGBoost'''
    lr = LogisticRegression(max_iter = 1000)
    rf = RandomForestClassifier()
    xgb = XGBClassifier()
    lr_acc = []
    rf_acc = []
    xgb_acc = []
    # repeat 3 times
    for i in range(3):
        num_sample = processed_embedding.impact.value_counts().min()
        emb_subset = processed_embedding.groupby('impact').sample(num_sample).reset_index() # subset to have equal samples in the two classes
        X_train, X_test, y_train, y_test = train_test_split(emb_subset.iloc[:, :64], emb_subset['impact'], test_size = 0.2)
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        
        lr_acc.append(round(lr.score(X_test, y_test), 2))
        rf_acc.append(round(rf.score(X_test, y_test), 2))
        xgb_predict = xgb.predict(X_test)
        xgb_acc.append(round(accuracy_score(y_test, xgb_predict), 2))

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            pickle.dump(lr, open(output_dir + f'lr_{emb_name}{i}.pkl', 'wb'))
            pickle.dump(rf, open(output_dir + f'rf_{emb_name}{i}.pkl', 'wb'))
            pickle.dump(xgb, open(output_dir + f'xgb_{emb_name}{i}.pkl', 'wb'))
            print(emb_name, 'model saved')
    if print_accuracy == True:
        print('lr average:', round(np.mean(lr_acc), 2), '; ',
              'rf average:', round(np.mean(rf_acc), 2), '; ', 
              'xgb_average:', round(np.mean(xgb_acc), 2))
    acc_df = pd.DataFrame({'LR':lr_acc, 'RF':rf_acc, 'XGB':xgb_acc})
    sns.boxplot(x = 'variable', y = 'value', data = pd.melt(acc_df))
    plt.ylim(0, 1)
    plt.title(output_dir)
    plt.ylabel('Accuracy')
    plt.xlabel('')
    plt.show()
    plt.close()
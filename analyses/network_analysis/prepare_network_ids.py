import pandas as pd

def prepare_net_ids():
    adj_df = pd.read_csv('./Data/Kapoor_adjacency.csv', index_col = 0)
    
    network_IDs = pd.Series(adj_df.columns)
    network_IDs.to_csv('./Data/network_analysis/network_IDs.csv')
    
if __name__ == '__main__':
    prepare_net_ids()
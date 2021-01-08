import argparse
import json
import os
import pandas as pd
from src.preproc.input import Input

def preproc(config_file, archive_path):
    Input('./Data')
    data_folder = Input.getPath()
    print("config_file: {} data_folder: {} archive_path: {}".format(config_file, data_folder, archive_path))
    result_path = os.path.join(data_folder, archive_path)
    config_path = os.path.join(data_folder, config_file)
    print("config_path: {}".format(config_path))

    with open(config_path) as json_data:
        config_json = json.load(json_data)

    adj_df = pd.read_csv(os.path.join(data_folder, config_json["provided_network_adjacency"]), index_col = 0)
    
    network_IDs = pd.Series(adj_df.columns)
    network_IDs.to_csv(os.path.join(result_path, config_json["network_ids"]))

    expression = pd.read_csv(os.path.join(data_folder, config_json["normalized_counts"]), sep = '\t')
    network_only_expression = expression[expression.id.isin(network_IDs)]
    network_only_expression.to_csv(os.path.join(result_path, config_json["network_only_expression"]), index = 0)
    network_only_expression_t = network_only_expression.T
    network_only_expression_t.drop('id', inplace=True)
    meta = pd.read_csv(os.path.join(data_folder, config_json["diagnostics"]))
    expression_meta = pd.merge(network_only_expression_t, meta, left_index = True, right_on = 'IID')
    expression_meta.to_csv(os.path.join(result_path, config_json["expression_with_metadata"]), index = 0)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="path to save results")
    args = parser.parse_args()
    preproc(args.config_file, args.archive_path)
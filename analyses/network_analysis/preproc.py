import argparse
import json
import os
import pandas as pd
from src.preproc.input import Input
from src.preproc.result import Result

def preproc(config_file, archive_path):
    Input('./Data')
    data_folder = Input.getPath()
    print("config_file: {} data_folder: {} archive_path: {}".format(config_file, data_folder, archive_path))
    Result(archive_path, overwrite=False)

    with open(config_file) as json_data:
        config_json = json.load(json_data)
    if ("skip_preproc" not in config_json) or (config_json["skip_preproc"] is False):
        meta = pd.read_csv(os.path.join(data_folder, config_json["diagnostics"]))
        expression = pd.read_csv(os.path.join(data_folder, config_json["normalized_counts"]), sep = '\t', index_col = 0)
        expression_meta = pd.merge(expression.T, meta, left_index = True, right_on = 'IID')
        expression_meta.to_csv(os.path.join(Result.getPath(), config_json["expression_with_metadata"]), index = 0)
    else:
        return None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="path to save results")
    args = parser.parse_args()
    preproc(args.config_file, args.archive_path)
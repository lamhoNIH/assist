import argparse
import json
import os
import pandas as pd
#from preproc.input import Input
#from preproc.result import Result

def preproc(config_file):
    print(f'inside preproc: {config_file}')
    with open(config_file) as json_data:
        config_json = json.load(json_data)
    if ("skip_preproc" not in config_json["parameters"]) or (json.loads(config_json["parameters"]["skip_preproc"].lower()) is False):
        meta = pd.read_csv(config_json["inputs"]["diagnostics"])
        expression = pd.read_csv(config_json["inputs"]["normalized_counts"], sep = '\t', index_col = 0)
        expression_meta = pd.merge(expression.T, meta, left_index = True, right_on = 'IID')
        expression_meta.to_csv(os.path.join(config_json["outputs"]["expression_with_metadata"]), index = 0)
    else:
        print("skip_preproc")
        return None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    args = parser.parse_args()
    preproc(args.config_file)
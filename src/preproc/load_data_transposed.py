import pandas as pd
import numpy as np
from pymongo import MongoClient

def load(tsv_data_file):
    df = pd.read_csv(tsv_data_file, sep='\t')#, index_col=0)
    df.update(df[['gid']].applymap('"{}"'.format))
    sdf = df[df.columns[0:3]]
    print(sdf)
    print(df.index.values)
    print(len(df.columns))
    tdf = df.T
    new_header = tdf.iloc[0]
    tdf = tdf[1:]
    #tdf.set_index("gid")
    tdf.columns = new_header
    print("headers after transpose: {}".format(len(tdf.columns)))
    print(tdf.head(5))
    tdf['pid'] = tdf.index
    client =  MongoClient("mongodb://localhost:27017")
    db = client["LINCS"]
    collection = db["samples"]
    data_dict = tdf.to_dict("records")
    collection.insert_many(data_dict)
    
if __name__ == '__main__':
    #load("/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data/LINCS L1000 (from GEO)/samples.tsv")
    load("../test/samples.tsv")
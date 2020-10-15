import sys
import pandas as pd
import numpy as np
from pymongo import MongoClient

def load(tsv_data_file):
    df = pd.read_csv(tsv_data_file, sep='\t')#, index_col=0)
    #df.update(df[['Entrez id']].applymap('"{}"'.format))
    sdf = df[df.columns[0:3]]
    print(sdf)
    print(len(sdf.columns))

    client =  MongoClient("mongodb://localhost:27017")
    db = client["LINCS"]
    collection = db["L1000"]
    print("preparing data_dict")
    data_dict = df.to_dict("records")
    print("loaded data_dict")
    collection.insert_many(data_dict)
    print("after insert_many")
    
def print_cell(gid, sid):
    client =  MongoClient("mongodb://localhost:27017")
    db = client["LINCS"]
    collection = db["samples"]
    matches = collection.find({'Entrez id':gid})
    print(matches.count())
    for m in matches:
        print(type(m))
        print("value: {}".format(m[sid]))
    
if __name__ == '__main__':
    #load("/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data/LINCS L1000 (from GEO)/samples.tsv")
    print_cell(5720, "ASG001_MCF7_24H:BRD-A06304526-001-01-0:0#4")
    print_cell(6009, "ASG001_MCF7_24H:BRD-A06304526-001-01-0:2")
    #load("../reference/L1000/" + sys.argv[1])
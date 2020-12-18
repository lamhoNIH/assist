import pandas as pd
from os import path
from sys import platform

def load():
    prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'
    comm_df = pd.read_csv(prefix + '/Shared drives/NIAAA_ASSIST/Data/eda_derived/network_louvain_default.csv')
    return comm_df

comm_df = load()
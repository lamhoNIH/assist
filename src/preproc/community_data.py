import os
import pandas as pd
from .input import Input

class CommunityData:
    __comm_df = None

    def get_comm_df():
        if CommunityData.__comm_df is None:
            CommunityData()
        return CommunityData.__comm_df
        
    def __init__(self):
        root_dir = Input.getPath()
        CommunityData.__comm_df = pd.read_csv(os.path.join(root_dir, 'eda_derived/network_louvain_default.csv'))
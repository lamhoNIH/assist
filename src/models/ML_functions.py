import pandas as pd
import numpy as np
from sys import platform
from ..preproc.deseq_data import deseq


def process_emb_for_ML(embedding_df, deseq = deseq):
    embedding_labeled_df = pd.merge(embedding_df, deseq, left_index = True, right_on = 'id')
    embedding_labeled_df['impact'] = 1
    embedding_labeled_df.loc[embedding_labeled_df['log2FoldChange'].between(-0.1, 0.1), 'impact'] = 0
    return embedding_labeled_df
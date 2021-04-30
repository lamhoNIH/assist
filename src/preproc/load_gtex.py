import pandas as pd
import numpy as np

def clean_gtex(gtex_file):
    gtex = pd.read_csv(gtex_file, skiprows = 13)
    gtex = gtex[20:].reset_index(drop = True)
    gtex_clean = gtex.replace('x', np.NaN)
    return gtex_clean